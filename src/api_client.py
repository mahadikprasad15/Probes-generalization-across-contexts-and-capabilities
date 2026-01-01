"""
API clients for fast prompt generation using external inference services.
Supports: Groq, Cerebras, OpenAI (and compatible endpoints)
"""
import os
import json
import time
from typing import Optional, List, Dict
from enum import Enum

class APIProvider(Enum):
    GROQ = "groq"
    CEREBRAS = "cerebras"
    OPENAI = "openai"
    TOGETHER = "together"

class LLMAPIClient:
    """
    Unified client for multiple LLM API providers.
    Handles retries, rate limiting, and provider-specific quirks.
    """

    def __init__(
        self,
        provider: APIProvider,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3
    ):
        self.provider = provider
        self.max_retries = max_retries

        # Get API key from argument or environment
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = self._get_api_key_from_env()

        if not self.api_key:
            raise ValueError(f"No API key provided for {provider.value}. Set via argument or environment variable.")

        # Set default models per provider
        self.model = model or self._get_default_model()

        # Import the appropriate client
        self._setup_client()

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variables."""
        env_map = {
            APIProvider.GROQ: "GROQ_API_KEY",
            APIProvider.CEREBRAS: "CEREBRAS_API_KEY",
            APIProvider.OPENAI: "OPENAI_API_KEY",
            APIProvider.TOGETHER: "TOGETHER_API_KEY"
        }
        return os.getenv(env_map.get(self.provider))

    def _get_default_model(self) -> str:
        """Get default model for each provider."""
        defaults = {
            APIProvider.GROQ: "llama-3.3-70b-versatile",
            APIProvider.CEREBRAS: "llama-3.3-70b",
            APIProvider.OPENAI: "gpt-4o-mini",
            APIProvider.TOGETHER: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        }
        return defaults[self.provider]

    def _setup_client(self):
        """Initialize the API client based on provider."""
        if self.provider == APIProvider.GROQ:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)

        elif self.provider == APIProvider.CEREBRAS:
            from cerebras.cloud.sdk import Cerebras
            self.client = Cerebras(api_key=self.api_key)

        elif self.provider == APIProvider.OPENAI:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)

        elif self.provider == APIProvider.TOGETHER:
            from openai import OpenAI
            # Together AI uses OpenAI-compatible API
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.together.xyz/v1"
            )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.8,
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        Send a chat completion request with automatic retries.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                return response.choices[0].message.content

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"API error (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"API error after {self.max_retries} attempts: {e}")
                    raise

        raise RuntimeError("Failed to get API response after retries")


def create_api_client(
    provider_name: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> LLMAPIClient:
    """
    Factory function to create an API client.

    Args:
        provider_name: Name of provider ('groq', 'cerebras', 'openai', 'together')
        api_key: Optional API key (otherwise reads from environment)
        model: Optional model name (otherwise uses provider default)

    Returns:
        Configured LLMAPIClient instance
    """
    provider_map = {
        "groq": APIProvider.GROQ,
        "cerebras": APIProvider.CEREBRAS,
        "openai": APIProvider.OPENAI,
        "together": APIProvider.TOGETHER
    }

    if provider_name.lower() not in provider_map:
        raise ValueError(f"Unknown provider: {provider_name}. Choose from: {list(provider_map.keys())}")

    provider = provider_map[provider_name.lower()]
    return LLMAPIClient(provider=provider, api_key=api_key, model=model)
