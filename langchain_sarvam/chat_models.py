from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, Literal, cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import get_pydantic_field_names, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self


class ChatSarvam(BaseChatModel):
    """Sarvam AI chat model integration for LangChain.

    Sarvam AI provides multilingual AI models with native support for 10+ 
    Indic languages including Hindi, Bengali, Telugu, Tamil, and more.

    Setup:
        Install ``langchain-sarvam`` and set your API key:

        .. code-block:: bash

            pip install langchain-sarvam
            export SARVAM_API_KEY="your-api-key"

    Key init args — completion params:
        model_name: Model name to use. Defaults to "sarvam-m".
        temperature: Sampling temperature between 0.0 and 2.0. Higher values
            make output more random. Defaults to 0.7.
        max_tokens: Maximum number of tokens to generate. If None, will use
            model's default maximum.
        top_p: Nucleus sampling parameter. Defaults to 1.0.
        n: Number of completions to generate. Must be 1 when streaming.
        stop: Stop sequences. Can be a string or list of strings.
        frequency_penalty: Penalize frequent tokens. Defaults to None.
        presence_penalty: Penalize new tokens. Defaults to None.
        reasoning_effort: Reasoning effort level. One of "low", "medium", "high".
        seed: Random seed for reproducibility.
        wiki_grounding: Enable wiki grounding. Defaults to None.

    Key init args — client params:
        sarvam_api_key: Sarvam AI API key. If not passed in will be read from env var SARVAM_API_KEY.
        request_timeout: Request timeout in seconds.
        streaming: Whether to stream responses. Defaults to False.
        http_client: Custom HTTP client for sync requests.
        http_async_client: Custom HTTP client for async requests.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_sarvam import ChatSarvam

            llm = ChatSarvam(
                model_name="sarvam-m",
                temperature=0.7,
                max_tokens=256,
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful assistant that speaks Hindi."),
                ("human", "What is the color of the sky?"),
            ]
            llm.invoke(messages)

        .. code-block:: python

            AIMessage(content='आसमान का रंग नीला होता है।', response_metadata={...})

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk.content, end="", flush=True)

    Async:
        .. code-block:: python

            await llm.ainvoke(messages)

        .. code-block:: python

            async for chunk in llm.astream(messages):
                print(chunk.content, end="", flush=True)

    Batch:
        .. code-block:: python

            llm.batch([messages1, messages2])

    Multilingual support:
        Sarvam AI natively supports 10+ Indic languages:

        .. code-block:: python

            # Hindi
            messages = [
                ("system", "talk in Hindi"),
                ("human", "Hello, how are you?"),
            ]
            response = llm.invoke(messages)
            print(response.content)  # Output in Hindi

    Response metadata:
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

            {
                'token_usage': {'completion_tokens': 12, 'prompt_tokens': 57, 'total_tokens': 69},
                'model_name': 'sarvam-m',
                'finish_reason': 'stop',
            }
    """

    # Client instances (internal use only)
    client: Any = Field(default=None, exclude=True, description="Internal Sarvam AI synchronous client instance.")
    async_client: Any = Field(default=None, exclude=True, description="Internal Sarvam AI asynchronous client instance.")

    # Model parameters
    model_name: str = Field(alias="model", description="Model name to use for chat completions. Defaults to 'sarvam-m'.")
    temperature: float | None = Field(default=None, description="Sampling temperature between 0.0 and 2.0. Higher values make output more random.")
    top_p: float | None = Field(default=None, description="Nucleus sampling parameter. Controls diversity via cumulative probability.")
    max_tokens: int | None = Field(default=None, description="Maximum number of tokens to generate. If None, uses model's default maximum.")
    n: int = Field(default=1, description="Number of completions to generate. Must be 1 when streaming is enabled.")
    stop: list[str] | str | None = Field(default=None, alias="stop_sequences", description="Stop sequences. Can be a string or list of strings.")

    # Advanced parameters
    frequency_penalty: float | None = Field(default=None, description="Penalizes frequent tokens to reduce repetition. Values between -2.0 and 2.0.")
    presence_penalty: float | None = Field(default=None, description="Penalizes new tokens to encourage topic diversity. Values between -2.0 and 2.0.")
    reasoning_effort: Literal["low", "medium", "high"] | None = Field(default=None, description="Reasoning effort level for the model.")
    seed: int | None = Field(default=None, description="Random seed for reproducible outputs.")
    wiki_grounding: bool | None = Field(default=None, description="Enable wiki grounding for factual responses.")

    # Additional model kwargs
    model_kwargs: dict[str, Any] = Field(default_factory=dict, description="Additional keyword arguments passed to the Sarvam AI API.")

    # Authentication and client configuration
    sarvam_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=lambda: secret_from_env("SARVAM_API_KEY", default=None),
        description="Sarvam AI API key. If not provided, reads from SARVAM_API_KEY environment variable."
    )
    request_timeout: float | None = Field(default=None, alias="timeout", description="Request timeout in seconds.")

    # HTTP client customization
    http_client: Any | None = Field(default=None, description="Custom HTTP client for synchronous requests.")
    http_async_client: Any | None = Field(default=None, description="Custom HTTP client for asynchronous requests.")

    # Streaming configuration
    streaming: bool = Field(default=False, description="Whether to stream responses. When True, enables real-time token streaming.")

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """Build extra model kwargs from unknown fields.

        This method processes fields that are not explicitly defined in the model
        and moves them to the model_kwargs dictionary for passing to the API.

        Args:
            values: Raw input values before validation.

        Returns:
            Processed values with extra fields moved to model_kwargs.

        Raises:
            ValueError: If a field is specified both explicitly and in model_kwargs.
        """
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                msg = f"Found {field_name} supplied twice."
                raise ValueError(msg)
            if field_name not in all_required_field_names:
                extra[field_name] = values.pop(field_name)
        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            msg = (
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                "Instead they were passed in as part of `model_kwargs` parameter."
            )
            raise ValueError(msg)
        values["model_kwargs"] = extra
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate the environment and initialize clients.

        This method performs validation of model parameters and initializes
        the Sarvam AI client instances for both sync and async operations.

        Returns:
            Self: The validated model instance.

        Raises:
            ValueError: If n < 1 or if streaming is enabled with n > 1.
            ImportError: If the sarvamai package is not installed.
            ValueError: If the API key is not provided.
        """
        if self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.n > 1 and self.streaming:
            raise ValueError("n must be 1 when streaming.")

        try:
            from sarvamai import AsyncSarvamAI, SarvamAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Could not import sarvamai python package. Please install it with `pip install sarvamai`."
            ) from exc

        client_params: dict[str, Any] = {
            "api_subscription_key": (
                self.sarvam_api_key.get_secret_value() if self.sarvam_api_key else None
            ),
            "timeout": self.request_timeout,
        }

        if client_params["api_subscription_key"] is None:
            raise ValueError(
                "Sarvam API key is not set. Set `sarvam_api_key` field or `SARVAM_API_KEY` env var."
            )

        if not self.client:
            sync_specific: dict[str, Any] = {}
            if self.http_client is not None:
                sync_specific["httpx_client"] = self.http_client
            self.client = SarvamAI(**client_params, **sync_specific).chat
        if not self.async_client:
            async_specific: dict[str, Any] = {}
            if self.http_async_client is not None:
                async_specific["httpx_client"] = self.http_async_client
            self.async_client = AsyncSarvamAI(**client_params, **async_specific).chat
        return self

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Return the secret environment variable names for this model.

        Returns:
            Dictionary mapping field names to environment variable names.
        """
        return {"sarvam_api_key": "SARVAM_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by LangChain.

        Returns:
            True, as ChatSarvam supports serialization.
        """
        return True

    @property
    def _llm_type(self) -> str:
        """Return the type identifier for this LLM.

        Returns:
            String identifier for LangChain's internal use.
        """
        return "sarvam-chat"

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get parameters for LangSmith tracing.

        Args:
            stop: Stop sequences to override the instance default.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            LangSmithParams object with tracing information.
        """
        # kwargs is unused but required by base class interface
        ls_params = LangSmithParams(
            ls_provider="sarvam",
            ls_model_name=self.model_name,
            ls_model_type="chat",
            ls_temperature=self.temperature,
            ls_max_tokens=self.max_tokens,
            ls_stop=stop or self.stop,
        )
        if isinstance(self.max_tokens, int):
            ls_params["ls_max_tokens"] = self.max_tokens
        if stop or self.stop:
            ls_stop: list[str] | None
            if stop is not None:
                ls_stop = stop
            elif isinstance(self.stop, list):
                ls_stop = self.stop
            elif isinstance(self.stop, str):
                ls_stop = [self.stop]
            else:
                ls_stop = None
            if ls_stop:
                ls_params["ls_stop"] = ls_stop
        return ls_params

    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for API calls.

        Returns:
            Dictionary of parameters to send to the Sarvam AI API.
        """
        # Sarvam SDK does not accept a 'model' parameter currently; default model is sarvam-m.
        params: dict[str, Any] = {
            "n": self.n,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.stop is not None:
            params["stop"] = self.stop
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        if self.seed is not None:
            params["seed"] = self.seed
        if self.wiki_grounding is not None:
            params["wiki_grounding"] = self.wiki_grounding
        if self.model_kwargs:
            params.update(self.model_kwargs)
        return params

    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: list[str] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Create message dictionaries and parameters for API calls.

        Args:
            messages: List of BaseMessage objects to convert.
            stop: Stop sequences to override instance defaults.

        Returns:
            Tuple of (message_dicts, params) for the API call.
        """
        params = self._default_params()
        if stop is not None:
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat completion synchronously.

        Args:
            messages: List of messages for the conversation.
            stop: Optional stop sequences.
            run_manager: Callback manager for run tracking.
            **kwargs: Additional keyword arguments.

        Returns:
            ChatResult containing the generated response.
        """
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        resp = self.client.completions(messages=message_dicts, **params)
        return self._create_chat_result(resp, params)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat completion asynchronously.

        Args:
            messages: List of messages for the conversation.
            stop: Optional stop sequences.
            run_manager: Async callback manager for run tracking.
            **kwargs: Additional keyword arguments.

        Returns:
            ChatResult containing the generated response.
        """
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        resp = await self.async_client.completions(messages=message_dicts, **params)
        return self._create_chat_result(resp, params)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completion responses synchronously.

        Args:
            messages: List of messages for the conversation.
            stop: Optional stop sequences.
            run_manager: Callback manager for run tracking.
            **kwargs: Additional keyword arguments.

        Yields:
            ChatGenerationChunk objects as they become available.
        """
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        for chunk in self.client.completions(messages=message_dicts, **params):
            processed_chunk = chunk
            if not isinstance(processed_chunk, dict):
                processed_chunk = processed_chunk.model_dump()  # type: ignore[attr-defined]
            if len(processed_chunk.get("choices", [])) == 0:
                continue
            choice = processed_chunk["choices"][0]
            message_chunk = _convert_chunk_to_message_chunk(
                processed_chunk, default_chunk_class
            )
            generation_info: dict[str, Any] = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
                if model_name := processed_chunk.get("model"):
                    generation_info["model_name"] = model_name
                if system_fingerprint := processed_chunk.get("system_fingerprint"):
                    generation_info["system_fingerprint"] = system_fingerprint
            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=generation_info or None
            )
            if run_manager:
                run_manager.on_llm_new_token(
                    generation_chunk.text, chunk=generation_chunk
                )
            yield generation_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream chat completion responses asynchronously.

        Args:
            messages: List of messages for the conversation.
            stop: Optional stop sequences.
            run_manager: Async callback manager for run tracking.
            **kwargs: Additional keyword arguments.

        Yields:
            ChatGenerationChunk objects as they become available.
        """
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        async for chunk in self.async_client.completions(
            messages=message_dicts, **params
        ):
            processed_chunk = chunk
            if not isinstance(processed_chunk, dict):
                processed_chunk = processed_chunk.model_dump()  # type: ignore[attr-defined]
            if len(processed_chunk.get("choices", [])) == 0:
                continue
            choice = processed_chunk["choices"][0]
            message_chunk = _convert_chunk_to_message_chunk(
                processed_chunk, default_chunk_class
            )
            generation_info: dict[str, Any] = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
                if model_name := processed_chunk.get("model"):
                    generation_info["model_name"] = model_name
                if system_fingerprint := processed_chunk.get("system_fingerprint"):
                    generation_info["system_fingerprint"] = system_fingerprint
            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=generation_info or None
            )
            if run_manager:
                await run_manager.on_llm_new_token(
                    token=generation_chunk.text, chunk=generation_chunk
                )
            yield generation_chunk

    def _create_chat_result(
        self, response: dict | BaseModel, params: Mapping[str, Any]
    ) -> ChatResult:
        """Create a ChatResult from the API response.

        Args:
            response: Raw response from the Sarvam AI API.
            params: Parameters used for the request (unused but required by interface).

        Returns:
            ChatResult containing the processed response.
        """
        # params is unused but required by base class interface
        generations: list[ChatGeneration] = []
        if not isinstance(response, dict):
            response = response.model_dump()  # type: ignore[attr-defined]
        token_usage = response.get("usage", {})
        for res in response.get("choices", []):
            message = _convert_dict_to_message(res["message"])  # type: ignore[index]
            if token_usage and isinstance(message, AIMessage):
                input_tokens = token_usage.get("prompt_tokens", 0)
                output_tokens = token_usage.get("completion_tokens", 0)
                message.usage_metadata = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": token_usage.get(
                        "total_tokens", input_tokens + output_tokens
                    ),
                }
            generation_info: dict[str, Any] = {
                "finish_reason": res.get("finish_reason")
            }
            gen = ChatGeneration(message=message, generation_info=generation_info)
            generations.append(gen)
        llm_output: dict[str, Any] = {}
        if token_usage:
            llm_output["token_usage"] = token_usage
        if model_name := response.get("model"):
            llm_output["model_name"] = model_name
        if system_fingerprint := response.get("system_fingerprint"):
            llm_output["system_fingerprint"] = system_fingerprint
        return ChatResult(generations=generations, llm_output=llm_output or None)


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    if isinstance(message, ChatMessage):
        return {"role": message.role, "content": message.content}
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AIMessage):
        content = message.content
        if isinstance(content, list):
            text_blocks = [
                block
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = text_blocks if text_blocks else ""
        return {"role": "assistant", "content": content}
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, FunctionMessage):
        return {"role": "function", "content": message.content, "name": message.name}
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    msg = f"Got unknown type {message}"
    raise TypeError(msg)


def _convert_chunk_to_message_chunk(
    chunk: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    choice = chunk["choices"][0]
    delta = cast("Mapping[str, Any]", choice.get("delta", {}))
    role = cast("str | None", delta.get("role"))
    content = cast("str", delta.get("content") or "")

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content)
    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    if role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=delta.get("name"))  # type: ignore[arg-type]
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=delta.get("tool_call_id"))  # type: ignore[arg-type]
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    return default_class(content=content)  # type: ignore[call-arg]


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""))
    if role == "assistant":
        return AIMessage(
            content=_dict.get("content", "") or "",
            response_metadata={"model_provider": "sarvam"},
        )
    if role == "system":
        return SystemMessage(content=_dict.get("content", ""))
    if role == "function":
        return FunctionMessage(content=_dict.get("content", ""), name=_dict.get("name"))  # type: ignore[arg-type]
    if role == "tool":
        return ToolMessage(
            content=_dict.get("content", ""), tool_call_id=_dict.get("tool_call_id")
        )  # type: ignore[arg-type]
    return ChatMessage(content=_dict.get("content", ""), role=cast("str", role))
