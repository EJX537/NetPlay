"""
Simple LiteLLM wrapper that provides a consistent interface for various LLM providers.
Replaces the need for provider-specific adapters.

Environment variables for different providers:
- OpenAI: OPENAI_API_KEY
- Gemini: GEMINI_API_KEY
- Anthropic: ANTHROPIC_API_KEY
"""
from typing import List
from langchain.schema import AIMessage, BaseMessage
import litellm
import os


class LiteLLMWrapper:
    """Wrapper around LiteLLM that provides LangChain-compatible interface."""

    def __init__(self, model: str, temperature: float = 0.0, max_retries: int = 0, **kwargs):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.kwargs = kwargs

        # Validate API keys are set for the provider
        self._validate_api_key()

    def get_num_tokens(self, content: str) -> int:
        """Estimate token count for content."""
        if content is None:
            return 0
        try:
            # Use litellm's token counter if available
            return litellm.token_counter(model=self.model, text=content)
        except Exception:
            # Fallback to heuristic
            return max(1, len(content) // 4)

    def predict_messages(self, messages: List[BaseMessage]) -> AIMessage:
        """
        Send messages to LiteLLM and return response.

        Args:
            messages: List of LangChain BaseMessage objects

        Returns:
            AIMessage with the response content
        """
        # Convert LangChain messages to LiteLLM format
        litellm_messages = []
        for msg in messages:
            role = self._get_role(msg)
            content = getattr(msg, 'content', str(msg))
            litellm_messages.append({"role": role, "content": content})

        # Call LiteLLM
        try:
            response = litellm.completion(
                model=self.model,
                messages=litellm_messages,
                temperature=self.temperature,
                **self.kwargs
            )

            # Extract content from response
            content = response.choices[0].message.content
            
            # Debug: Print first response to help diagnose issues
            if not hasattr(self, '_first_response_printed'):
                print(f"\n=== DEBUG: First LLM Response ===")
                print(f"Model: {self.model}")
                print(f"Response content: {content[:500] if content else 'None'}")
                print(f"=== END DEBUG ===\n")
                self._first_response_printed = True
            
            return AIMessage(content=content)

        except Exception as e:
            # Re-raise with context
            raise Exception(f"LiteLLM call failed: {str(e)}") from e

    def _get_role(self, message: BaseMessage) -> str:
        """Convert LangChain message type to chat role."""
        msg_type = type(message).__name__.lower()

        if 'system' in msg_type:
            return 'system'
        elif 'human' in msg_type or 'user' in msg_type:
            return 'user'
        elif 'ai' in msg_type or 'assistant' in msg_type:
            return 'assistant'
        else:
            # Default to user for unknown types
            return 'user'

    def _validate_api_key(self):
        """Check if required API key is set for the model provider."""
        provider_keys = {
            'gpt': 'OPENAI_API_KEY',
            'claude': 'ANTHROPIC_API_KEY',
            'gemini': 'GEMINI_API_KEY',
            'command': 'COHERE_API_KEY',
            'mistral': 'MISTRAL_API_KEY',
        }

        # Determine provider from model name
        model_lower = self.model.lower()
        required_key = None

        for prefix, key in provider_keys.items():
            if prefix in model_lower:
                required_key = key
                break

        # Check if key is set
        if required_key and not os.getenv(required_key):
            print(f"Warning: {required_key} environment variable not set for model '{self.model}'")
            print(f"Set it with: export {required_key}='your-api-key'")
            print(f"See: https://docs.litellm.ai/docs/providers")
