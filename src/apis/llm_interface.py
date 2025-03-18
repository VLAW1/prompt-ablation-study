"""
Interface for calling APIs for LLM responses.
"""

import os
import logging
import time
import json

import pandas as pd

import anthropic
from anthropic.types.message_create_params import (
    MessageCreateParamsNonStreaming,
)
from anthropic.types.messages.batch_create_params import Request
from google import genai
from google.genai import types
import openai

from src.apis.mock_client import MockClient


log = logging.getLogger(__name__)


class LLMInterface:
    """
    Interface for interacting with language models.
    """

    def __init__(self, model_config: dict[str, str | int | float]):
        """
        Initialize the model interface.

        Parameters
        ----------
        model_config : dict[str, str | int | float]
            Configuration for the model request
        """
        self.provider = model_config['provider']
        self.model_name = model_config['model']
        self.api_key = model_config['api_key']
        self.temperature = model_config.get('temperature', 0.7)
        self.max_tokens = model_config.get('max_tokens', 1024)

        self._initialize_client(self.provider)

    def _initialize_client(self, model_type: str) -> None:
        """
        Initialize the client based on the model type.

        Parameters
        ----------
        model_type : str
            The type of model to initialize

        Raises
        ------
        ValueError
            If an unsupported model type is provided
        """
        if model_type == 'mock':
            self.client = MockClient()
        elif model_type == 'openai':
            self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        elif model_type == 'anthropic':
            self.client = anthropic.Anthropic(
                api_key=os.getenv('ANTHROPIC_API_KEY')
            )
        elif model_type == 'google':
            self.client = genai.GoogleAPI(api_key=os.getenv('GOOGLE_API_KEY'))
        else:
            raise ValueError(f'Unsupported model type: {model_type}')

    def query(self, prompt: str) -> str:
        """
        Query the model with a prompt.

        Parameters
        ----------
        prompt : str
            The prompt to send to the model

        Returns
        -------
        str
            The model's response
        """

        if self.provider == 'mock':
            return self.client.query(prompt)

        elif self.provider == 'openai':
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content

        elif self.provider == 'anthropic':
            response = self.client.messages.create(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.content[0].text

        elif self.provider == 'google':
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                ),
            )
            return response.text

    def make_batch_request(
        self, prompts: pd.DataFrame, data_dir: str = ''
    ) -> tuple[str, str]:
        """
        Make a batch request and return the path to the responses.

        Parameters
        ----------
        prompts : pd.DataFrame
            A DataFrame containing the prompts to send to the model.
        response_path : str
            The path to save the responses to.

        Returns
        -------
        tuple[str, str]
            A tuple containing the batch ID and the path to the responses.

        Raises
        ------
        NotImplementedError
            If the provider does not support batch requests.
        """
        if self.provider != 'anthropic':
            raise NotImplementedError(
                f'Batch requests are only implemented for Anthropic, not {self.provider}'
            )

        def create_request(request_id: str, prompt: str) -> Request:
            """Helper function to create a request for a batch request."""
            request = Request(
                custom_id=request_id,
                params=MessageCreateParamsNonStreaming(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    messages=[
                        {
                            'role': 'user',
                            'content': prompt,
                        }
                    ],
                ),
            )
            return request

        log.info(f'Creating batch request for {len(prompts)} prompts')
        batch = self.client.messages.batches.create(
            requests=[
                create_request(row['request_id'], row['prompt'])
                for _, row in prompts.iterrows()
            ]
        )
        batch_id = batch.id
        log.info(f'Batch request created with ID: {batch.id}')

        time.sleep(60)
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            if batch.processing_status == 'ended':
                log.info(f'Batch {batch_id} completed.')
                break

            log.info(f'Batch {batch_id} is still processing...')
            time.sleep(60)

        response_path = f'batch_results_{batch_id}.jsonl'
        if data_dir != '':
            response_path = os.path.join(data_dir, response_path)

        with open(response_path, 'w') as f:
            for result in self.client.messages.batches.results(batch_id):
                f.write(json.dumps(result.model_dump()) + '\n')
        log.info(f'Batch results saved to {response_path}')

        return batch_id, response_path

    def retrieve_and_save_batch_results(self, batch_id: str) -> str:
        """
        Retrieve the results of a batch request.

        Parameters
        ----------
        batch_id : str
            The ID of the batch request

        Returns
        -------
        str
            The path to the results

        Raises
        ------
        NotImplementedError
            If the provider does not support batch requests
        """
        if self.provider != 'anthropic':
            raise NotImplementedError(
                f'Batch requests are only implemented for Anthropic, not {self.provider}'
            )

        with open(f'batch_results_{batch_id}.jsonl', 'w') as f:
            for result in self.client.messages.batches.results(batch_id):
                f.write(json.dumps(result.model_dump()) + '\n')
        log.info(f'Batch results saved to batch_results_{batch_id}.jsonl')

        return batch_id
