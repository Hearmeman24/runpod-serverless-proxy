# Importing necessary libraries
import logging
import os
import traceback

from fastapi import FastAPI, APIRouter, Request, HTTPException
from runpod_serverless import ApiConfig, RunpodServerlessCompletion, Params, RunpodServerlessEmbedding
from fastapi.responses import StreamingResponse, JSONResponse, Response
import json, time
from uvicorn import Config, Server
from pathlib import Path

# Initializing variables
model_data = {
    "object": "list",
    "data": []
}

configs = []
logger = logging.getLogger("proxy")

def _exc_to_payload(e: Exception):
    """Turn an exception (incl. requests/httpx HTTP errors) into a JSON body + status."""
    status = 500
    body = {
        "error": "proxy_error",
        "type": e.__class__.__name__,
        "message": str(e),
    }

    # If this came from requests/httpx, try to include upstream status/body
    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            upstream_body = resp.json()
        except Exception:
            upstream_body = getattr(resp, "text", "")[:4000]  # avoid huge logs
        status = getattr(resp, "status_code", status) or status
        body.update({
            "upstream_status": status,
            "upstream_body": upstream_body,
        })

    # Optional debug traceback (enable with DEBUG_ERRORS=1)
    if os.getenv("DEBUG_ERRORS", "0") == "1":
        body["traceback"] = traceback.format_exc(limit=25).splitlines()

    return status, body

def run(config_path: str, host: str = "127.0.0.1", port: int = 3000):
    if config_path:
        config_dict = load_config(config_path)  # function to load your config file

        for config in config_dict["models"]:
            config_model = {
                "url": f"https://api.runpod.ai/v2/{config['endpoint']}",
                "api_key": config_dict["api_key"],
                "model": config["model"],
                **({"timeout": config["timeout"]} if config.get("timeout") is not None else {}),
                **({"use_openai_format": config["use_openai_format"]} if config.get("use_openai_format") is not None else {}),
                **({"batch_size": config["batch_size"]} if config.get("batch_size") is not None else {}),
            }
            configs.append(ApiConfig(**config_model))
        for api in configs: print(api)

        model_data["data"] = [{"id": config["model"], 
            "object": "model", 
            "created": int(time.time()), 
            "owned_by": "organization-owner"} for config in config_dict["models"]]
        config = Config(
            app=app,
            host=config_dict.get("host", host),
            port=config_dict.get("port", port),
            log_level=config_dict.get("log_level", "info"),
        )
    else:
        config = Config(
            app=app,
            host=host,
            port=port,
            log_level="info",
        )
    server = Server(config=config)
    server.run()

def load_config(config_path):
    # Fixed: Use the parameter instead of args.config
    config_path = Path(config_path)
    with open(config_path) as f:
        return json.load(f)

# Function to get configuration by model name
def get_config_by_model(model_name):
    for config in configs:
        if config.model == model_name:
            return config

# Fixed function to format the response data
def format_response(data, model_name="gpt-3.5-turbo-instruct"):
    output_chunks = data.get("output", [])
    
    # Concatenate all tokens from all chunks
    all_tokens = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    for chunk in output_chunks:
        choices = chunk.get("choices", [])
        if choices:
            tokens = choices[0].get("tokens", [])
            all_tokens.extend(tokens)
            
            # Aggregate usage information
            usage = chunk.get("usage", {})
            if "input" in usage:
                total_input_tokens = max(total_input_tokens, usage.get("input", 0))
                total_output_tokens = usage.get("output", 0)  # Use the last chunk's output count

    full_text = "".join(all_tokens)
    
    openai_like_response = {
        'id': data['id'],
        'object': 'text_completion',
        'created': int(time.time()),
        'model': model_name,
        'system_fingerprint': "fp_44709d6fcb",
        'choices': [
            {
                'index': 0,
                'text': full_text,
                'logprobs': None,
                'finish_reason': 'stop' if data['status'] == 'COMPLETED' else 'length'
            }
        ],
        'usage': {
            'prompt_tokens': total_input_tokens,
            'completion_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens
        }
    }

    return openai_like_response

# Creating API router
router = APIRouter()

params = Params()

# API endpoint for chat completions
@router.post('/chat/completions')
async def request_chat(request: Request):
    try:
        data = await request.json()
        model = data.get("model")
        if not model:
            return JSONResponse(status_code=400, content={"detail": "Missing model in request."})

        api = get_config_by_model(model)
        payload = data.get("messages")

        params_dict = params.dict()
        params_dict.update(data)
        new_params = Params(**params_dict)
        runpod: RunpodServerlessCompletion = RunpodServerlessCompletion(api=api, params=new_params)

        # Non-streaming
        if not data.get("stream"):
            response = get_chat_synchronous(runpod, payload, model)
            # If your helper returns a requests.Response, pass it through verbatim:
            if hasattr(response, "status_code") and hasattr(response, "content"):
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    media_type=response.headers.get("content-type", "application/json"),
                )
            # Otherwise assume it's already a FastAPI response:
            return response

        # Streaming
        stream_data = get_chat_asynchronous(runpod, payload)

        # Wrap the generator so stream errors become a single JSON frame instead of killing the connection
        async def safe_stream(gen):
            try:
                async for chunk in gen:
                    yield chunk
            except Exception as e:
                status, body = _exc_to_payload(e)
                # Send one JSON error event and end the stream
                yield f"data: {json.dumps({'stream_error': body, 'status': status})}\n\n".encode()

        response = StreamingResponse(content=safe_stream(stream_data), media_type="text/event-stream")
        # Keep your explicit iterator if you need it:
        response.body_iterator = safe_stream(stream_data).__aiter__()
        return response

    except Exception as e:
        logger.exception("request_chat failed")
        status, body = _exc_to_payload(e)
        return JSONResponse(status_code=status, content=body)

# API endpoint for completions
@router.post('/completions')
async def request_prompt(request: Request):
    try:
        data = await request.json()
        model = data.get("model")
        if not model:
            return JSONResponse(status_code=400, content={"detail": "Missing model in request."})
        payload = data.get("prompt")[0]
        api = get_config_by_model(model)
        
        params_dict = params.dict()
        params_dict.update(data)
        new_params = Params(**params_dict)
        runpod: RunpodServerlessCompletion = RunpodServerlessCompletion(api=api, params=new_params)
        return get_synchronous(runpod, payload, model)
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

# API endpoint for embeddings
@router.post('/embeddings')
async def request_embeddings(request: Request):
    try:
        data = await request.json()
        model = data.get("model")
        if not model:
            return JSONResponse(status_code=400, content={"detail": "Missing model in request."})
        payload = data.get("input")
        api = get_config_by_model(model)
        runpod: RunpodServerlessEmbedding = RunpodServerlessEmbedding(api=api)
        return get_embedding(runpod, payload)
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

# Fixed function to get chat synchronously
def get_chat_synchronous(runpod, chat, model_name="gpt-3.5-turbo"):
    # Generate a response from the runpod
    response = runpod.generate(chat)
    # Check if the response is not cancelled
    if response['status'] != "CANCELLED":
        # Process all output chunks, not just the first one
        output_chunks = response.get("output", [])
        
        # Concatenate all tokens from all chunks
        all_tokens = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        for chunk in output_chunks:
            choices = chunk.get("choices", [])
            if choices:
                tokens = choices[0].get("tokens", [])
                all_tokens.extend(tokens)
                
                # Aggregate usage information
                usage = chunk.get("usage", {})
                if "input" in usage:
                    total_input_tokens = max(total_input_tokens, usage.get("input", 0))
                    total_output_tokens = usage.get("output", 0)  # Use the last chunk's output count
        
        # Create OpenAI-compatible response
        full_text = "".join(all_tokens)
        data = {
            'id': response['id'],
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': model_name,
            'choices': [
                {
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': full_text
                    },
                    'finish_reason': 'stop' if response['status'] == 'COMPLETED' else 'length'
                }
            ],
            'usage': {
                'prompt_tokens': total_input_tokens,
                'completion_tokens': total_output_tokens,
                'total_tokens': total_input_tokens + total_output_tokens
            }
        }
    else:
        # If the request is cancelled, raise an exception
        raise HTTPException(status_code=408, detail="Request timed out.")
    return data

# Fixed streaming function that handles RunPod's actual streaming format
async def get_chat_asynchronous(runpod, chat, model_name="gpt-3.5-turbo"):
    try:
        async for chunk in runpod.stream_generate(chat):
            # Handle dict chunks (the actual RunPod streaming format)
            if isinstance(chunk, dict):
                # Handle the RunPod streaming format: {"status": "IN_PROGRESS", "stream": [...]}
                if "stream" in chunk and isinstance(chunk["stream"], list):
                    for stream_item in chunk["stream"]:
                        if "output" in stream_item:
                            output = stream_item["output"]
                            choices = output.get("choices", [])
                            if choices and "tokens" in choices[0]:
                                tokens = choices[0]["tokens"]
                                for token in tokens:
                                    sse_message = {
                                        "id": f"chatcmpl-{int(time.time())}",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": model_name,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {
                                                    "content": token
                                                },
                                                "finish_reason": None
                                            }
                                        ]
                                    }
                                    yield f"data: {json.dumps(sse_message)}\n\n".encode("utf-8")
                
                # Send finish message if completed
                if chunk.get("status") == "COMPLETED":
                    final_message = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }
                        ]
                    }
                    yield f"data: {json.dumps(final_message)}\n\n".encode("utf-8")
                    break

    except Exception as e:
        print(f"Streaming error: {e}")
    
    # Always send [DONE] at the end
    yield b"data: [DONE]\n\n"

# Fixed function to get synchronous response
def get_synchronous(runpod, prompt, model_name="gpt-3.5-turbo-instruct"):
    # Generate a response from the runpod
    response = runpod.generate(prompt)
    # Check if the response is not cancelled
    if(response['status'] != "CANCELLED"):
        # Format the response with the correct model name
        data = format_response(response, model_name)
    else:
        # If the request is cancelled, raise an exception
        raise HTTPException(status_code=408, detail="Request timed out.")
    return data

# Function to get embeddings
def get_embedding(runpod, embedding):
    # Generate a response from the runpod
    response = runpod.generate(embedding)
    # Check if the response is not cancelled
    if(response['status'] != "CANCELLED"):
        # Format the response
        data = response["output"]
    else:
        # If the request is cancelled, raise an exception
        raise HTTPException(status_code=408, detail="Request timed out.")
    return data

# Function to prepare chat message for SSE
def prepare_chat_message_for_sse(stream_chunk_list: list) -> str:
    generated_text = ""
    for chunk in stream_chunk_list:
        choices = chunk.get("choices", [])
        if not choices:
            continue
        tokens = choices[0].get("tokens", [])
        generated_text += "".join(tokens)

    return json.dumps({
        "choices": [
            {
                "delta": {
                    "content": generated_text
                },
                "finish_reason": "stop"
            }
        ]
    })

# Create a FastAPI application
app = FastAPI()

# Include the router in the application
app.include_router(router)

# Endpoint to list all models
@app.get("/models")
async def list_models():
    return model_data

# Endpoint to get a specific model
@app.get("/models/{model_id}")
async def get_model(model_id):
    # Function to find a model by id
    def find_model(models, id):
        return next((model for model in models['data'] if model['id'] == id), None)
    # Return the found model
    return find_model(model_data, model_id)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", type=str, default=None)
    parser.add_argument("-e", "--endpoint", help="API endpoint", type=str, default=None)
    parser.add_argument("-k", "--api_key", help="API key", type=str, default=None)
    parser.add_argument("-m", "--model", help="Model", type=str, default=None)
    parser.add_argument("-t", "--timeout", help="Timeout", type=int, default=None)
    parser.add_argument("-o", "--use_openai_format", help="Use OpenAI format", type=bool, default=None)
    parser.add_argument("-b", "--batch_size", help="Batch size", type=int, default=None)
    parser.add_argument("--host", help="Host", type=str, default="127.0.0.1")
    parser.add_argument("--port", help="Port", type=int, default=3000)
    args = parser.parse_args()

    if args.config:
        run(args.config)
    else:
        config_model = {
            "url": f"https://api.runpod.ai/v2/{args.endpoint}",
            "api_key": args.api_key,
            "model": args.model,
            **({"timeout": args.timeout} if args.timeout is not None else {}),
            **({"use_openai_format": args.use_openai_format} if args.use_openai_format is not None else {}),
            **({"batch_size": args.batch_size} if args.batch_size is not None else {}),
        }
        configs.append(ApiConfig(**config_model))
        run(None, host=args.host, port=args.port)