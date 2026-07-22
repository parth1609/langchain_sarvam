# langchain-sarvam

## Overview

### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/sarvam) | Downloads | Version |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatSarvam](https://python.langchain.com/api_reference/sarvam/chat_models/langchain_sarvam.chat_models.ChatSarvam.html) | [langchain-sarvam](https://python.langchain.com/api_reference/sarvam/index.html) | ❌ | beta | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-sarvam?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-sarvam?style=flat-square&label=%20) |

### Model features

| [Tool calling](/oss/langchain/tools) | [Structured output](/oss/langchain/structured-output) | JSON mode | [Image input](/oss/langchain/messages#multimodal) | Audio input | Video input | [Token-level streaming](/oss/langchain/streaming#llm-tokens) | Native async | [Token usage](/oss/langchain/models#token-usage) | [Logprobs](/oss/langchain/models#log-probabilities) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |


Integration package connecting Sarvam AI chat completions with LangChain.

## Installation
 
```bash
pip install langchain-sarvam
```

Or :

```bash
uv add langchain-sarvam
```

## Setup

Set your Sarvam AI API key in your environment variables:

```bash
export SARVAM_API_KEY="your-api-key"
```

Or pass it in code:

```python
import os
from langchain_sarvam import ChatSarvam

llm = ChatSarvam(model="sarvam-30b", sarvam_api_key=os.getenv("SARVAM_API_KEY"))
```

## Usage

### Basic Usage

```python
from langchain_sarvam import ChatSarvam

llm = ChatSarvam(model="sarvam-30b", temperature=0.2, max_tokens=128)
resp = llm.invoke([("system", "You are helpful"), ("human", "Hello!")])
print(resp.content)
```

### Tool Calling

Bind Python functions decorated with `@tool` to `ChatSarvam`. The model intelligently selects and formats arguments for the appropriate tool.

```python
from langchain_core.tools import tool
from langchain_sarvam import ChatSarvam

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"32°C, Sunny in {city}"

@tool
def search_restaurants(city: str, cuisine: str) -> str:
    """Search for top-rated restaurants by cuisine type in a city."""
    return f"Top {cuisine} spots in {city}: Royal Kitchen"

# Initialize model and bind tools
llm = ChatSarvam(model="sarvam-30b", temperature=0)
model_with_tools = llm.bind_tools([get_weather, search_restaurants])

# Model generates tool call requests
response = model_with_tools.invoke("What's the weather in Mumbai?")
print(response.tool_calls)
# Output: [{'name': 'get_weather', 'args': {'city': 'Mumbai'}, 'id': '...', 'type': 'tool_call'}]
```

### Structured Output

Extract formatted Pydantic objects or JSON using `.with_structured_output()`. Supports `function_calling`, `json_schema`, and `json_mode`.

```python
from pydantic import BaseModel, Field
from langchain_sarvam import ChatSarvam

class AnswerWithJustification(BaseModel):
    """An answer along with justification."""
    answer: str = Field(description="The concise answer")
    justification: str = Field(description="Justification for the answer")

llm = ChatSarvam(model="sarvam-30b", temperature=0)

# Wrap LLM with structured output schema
structured_llm = llm.with_structured_output(AnswerWithJustification)

result = structured_llm.invoke("What weighs more, a pound of bricks or a pound of feathers?")
print("Answer:", result.answer)
print("Justification:", result.justification)
```

### Agent Integration (`create_agent`)

Combine Tool Calling and Structured Output into an autonomous agent loop:

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_sarvam import ChatSarvam

class AnswerWithJustification(BaseModel):
    answer: str = Field(description="The concise answer")
    justification: str = Field(description="Justification for the answer")

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"32°C, Sunny in {city}"

llm = ChatSarvam(model="sarvam-30b", temperature=0)

# Create agent with tools and structured response format
agent = create_agent(
    model=llm,
    tools=[get_weather],
    response_format=AnswerWithJustification,
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "What is the weather in Mumbai?"}]
})

# Access structured Pydantic object directly
result = response["structured_response"]
print("Answer:", result.answer)
print("Justification:", result.justification)
```

### Batch Processing

```python
from langchain_sarvam import ChatSarvam
from langchain_core.messages import HumanMessage

chat = ChatSarvam(model="sarvam-30b")

# Batch processing - use list of message lists
messages = [
    [HumanMessage(content="Tell me a joke")],
    [HumanMessage(content="What's the weather like?")]
]

responses = chat.batch(messages)
for response in responses:
    print(response.content)
```

### Using generate() Method

```python
from langchain_sarvam import ChatSarvam
from langchain_core.messages import HumanMessage

chat = ChatSarvam(model="sarvam-30b")

# generate() expects a list of message lists
inputs = [
    [HumanMessage(content="Tell me a joke with emojis only")],
    [HumanMessage(content="What's the weather like?")]
]

result = chat.generate(inputs)
for generation_list in result.generations:
    # generation_list is a list of ChatGeneration objects
    for generation in generation_list:
        print(generation.message.content)
```

### Streaming

```python
from langchain_sarvam import ChatSarvam

for chunk in ChatSarvam(model="sarvam-30b", streaming=True).stream("Tell me a joke"):
    print(chunk.content, end="")
```
