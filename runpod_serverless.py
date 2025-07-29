import time, json
import asyncio
from typing import Any, Dict, List, Mapping, Optional, AsyncIterator, Union
from pydantic import BaseModel
import requests
import aiohttp
from datetime import datetime


# Pydantic model for API configuration
class ApiConfig(BaseModel):
    url: str
    api_key: str
    model: str
    use_openai_format: int = 1
    timeout: int = 150
    batch_size: int = 30


# Pydantic model for parameters
class Params(BaseModel):
    max_tokens: int = 16
    n: int = 1
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop: Union[None, str, List[str]] = None
    seed: Optional[int] = None
    logprobs: Optional[int] = None


# Class for interacting with Runpod API
class RunpodServerlessCompletion:
    def __init__(self, api: ApiConfig, params: Params):
        self.api_key = api.api_key
        self.url = api.url
        self.model = api.model
        self.active_request_id = None
        self.params = params
        self.timeout = api.timeout
        self.use_openai_format = api.use_openai_format
        self.batch_size = api.batch_size
        self.extra_body = {}  # Initialize extra_body

    # Function to get the base URL for the request
    def _request_base_url(self) -> str:
        return self.url

    # Function to get the headers for the request
    def _request_headers(self) -> Mapping[str, str]:
        return {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": self.api_key,
        }

    # Function to perform a POST request to a given URL with a JSON payload
    def _post_request(self, url: str, json_data: Dict[str, Any]) -> Dict[str, Any]:
        headers = self._request_headers()
        response = requests.post(url, headers=headers, json=json_data)
        return response.json()

    # Function to perform a GET request to a given URL
    def _get_request(self, url: str) -> Dict[str, Any]:
        headers = self._request_headers()
        response = requests.get(url, headers=headers)
        return response.json()

    # Function to prepare the input for the request
    def _prepare_input(self, payload: Any, stream: bool = False, batch_size: int = None) -> Dict[str, Any]:
        if batch_size is None:
            batch_size = self.batch_size

        # Start with base sampling params
        sampling_params = self.params.model_dump()  # Use model_dump instead of deprecated dict()

        # Add extra_body parameters to sampling_params for vLLM worker
        if hasattr(self, 'extra_body') and self.extra_body:
            # For runpod/worker-vllm, chat_template_kwargs should be in sampling_params
            if 'chat_template_kwargs' in self.extra_body:
                sampling_params["chat_template_kwargs"] = self.extra_body["chat_template_kwargs"]
                print(f"Added chat_template_kwargs to sampling_params: {self.extra_body['chat_template_kwargs']}")

            # Add any other extra_body params to sampling_params as well
            for key, value in self.extra_body.items():
                if key != 'chat_template_kwargs':  # Already handled above
                    sampling_params[key] = value

            print(f"Final sampling_params: {sampling_params}")

        input = {
            "sampling_params": sampling_params,
            "stream": stream,
            "use_openai_format": self.use_openai_format,
            "batch_size": batch_size
        }

        if isinstance(payload, str):
            input["prompt"] = payload
        else:
            input["messages"] = payload

        return input

    # This function generates a request and waits for its completion or cancellation.
    # If the request takes more than the specified timeout, it cancels the request.
    def generate(self, payload) -> Dict[str, Any]:
        # Prepare the input data for the request
        input_data = self._prepare_input(payload)

        print(f"Sending request with input: {json.dumps(input_data, indent=2)}")

        response = self._post_request(f"{self._request_base_url()}/run", {"input": input_data})

        self.active_request_id = response["id"]

        start_time = time.time()
        while True:
            # Get the status of the request
            # If the request is completed or cancelled, log the metrics and break the loop
            response = self._get_request(f"{self._request_base_url()}/status/{self.active_request_id}")

            if response["status"] in ["COMPLETED", "CANCELLED"]:
                break

            # Calculate the elapsed time
            # If the elapsed time is more than the timeout, cancel the request and log the metrics
            elapsed_time = time.time() - start_time
            if elapsed_time > self.timeout:
                response = self.cancel_requests()
                print("Request timed out.")
                break
            # Wait for one second before checking the request status again
            time.sleep(1)

        return response

    # This function generates a request and yields its status and data in real time.
    # If the request takes more than the specified timeout, it cancels the request.
    async def stream_generate(self, payload) -> AsyncIterator[Dict[str, Any]]:

        input_data = self._prepare_input(payload, stream=True, batch_size=3)

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self._request_base_url()}/run", json={"input": input_data},
                                    headers=self._request_headers()) as response:
                response_data = await response.json()

                self.active_request_id = response_data["id"]
                data: Dict[str, Any] = {"status": "IN_QUEUE"}

                start_time = time.time()
                while data["status"] != "COMPLETED" and data["status"] != "CANCELLED":
                    try:
                        # Get the status and data of the request in real time
                        async with session.get(f"{self._request_base_url()}/stream/{self.active_request_id}",
                                               headers=self._request_headers()) as response:
                            async for chunk in response.content:
                                data = json.loads(chunk.decode('utf-8'))
                                # If there is stream data and the request is not completed, yield the data
                                if len(data.get("stream", [])) >= 1 and data["status"] != "COMPLETED":
                                    yield data
                        # Calculate the elapsed time
                        # If the elapsed time is more than the timeout, cancel the request and log the metrics
                        elapsed_time = time.time() - start_time

                        if elapsed_time > self.timeout:
                            response = self.cancel_requests()
                            print("Request timed out.")
                            yield response
                            break
                    except asyncio.TimeoutError:
                        print("Request timed out.")
                        break

    # This function cancels the active request.
    def cancel_requests(self) -> Optional[requests.Response]:
        # If there is no active request, return None
        if not self.active_request_id:
            return None
        # Post a cancellation request and return the response
        return self._post_request(f"{self._request_base_url()}/cancel/{self.active_request_id}", {})


# Class for interacting with Runpod API
class RunpodServerlessEmbedding:
    def __init__(self, api: ApiConfig):
        self.api_key = api.api_key
        self.url = api.url
        self.model = api.model
        self.active_request_id = None
        self.timeout = api.timeout

    # Function to get the base URL for the request
    def _request_base_url(self) -> str:
        return self.url

    # Function to get the headers for the request
    def _request_headers(self) -> Mapping[str, str]:
        return {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": self.api_key,
        }

    # Function to perform a POST request to a given URL with a JSON payload
    def _post_request(self, url: str, json_data: Dict[str, Any]) -> Dict[str, Any]:
        headers = self._request_headers()
        response = requests.post(url, headers=headers, json=json_data)
        return response.json()

    # Function to perform a GET request to a given URL
    def _get_request(self, url: str) -> Dict[str, Any]:
        headers = self._request_headers()
        response = requests.get(url, headers=headers)
        return response.json()

    # This function generates a request and waits for its completion or cancellation.
    # If the request takes more than the specified timeout, it cancels the request.
    def generate(self, input_data) -> Dict[str, Any]:
        # Prepare the input data for the request
        print(input_data)
        response = self._post_request(f"{self._request_base_url()}/run", {"input": {"sentences": input_data}})
        self.active_request_id = response["id"]

        start_time = time.time()
        while True:
            # Get the status of the request
            # If the request is completed or cancelled, log the metrics and break the loop
            response = self._get_request(f"{self._request_base_url()}/status/{self.active_request_id}")

            if response["status"] in ["COMPLETED", "CANCELLED"]:
                break

            # Calculate the elapsed time
            # If the elapsed time is more than the timeout, cancel the request and log the metrics
            elapsed_time = time.time() - start_time
            if elapsed_time > self.timeout:
                response = self.cancel_requests()
                print("Request timed out.")
                break
            # Wait for one second before checking the request status again
            time.sleep(1)

        return response

    # This function cancels the active request.
    def cancel_requests(self) -> Optional[requests.Response]:
        # If there is no active request, return None
        if not self.active_request_id:
            return None
        # Post a cancellation request and return the response
        return self._post_request(f"{self._request_base_url()}/cancel/{self.active_request_id}", {})