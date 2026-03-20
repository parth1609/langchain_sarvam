# langchain-sarvam

## Overview

### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/sarvam) | Downloads | Version |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatSarvam](https://python.langchain.com/api_reference/sarvam/chat_models/langchain_sarvam.chat_models.ChatSarvam.html) | [langchain-sarvam](https://python.langchain.com/api_reference/sarvam/index.html) | ❌ | beta | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-sarvam?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-sarvam?style=flat-square&label=%20) |

### Model features

| [Tool calling](/oss/langchain/tools) | [Structured output](/oss/langchain/structured-output) | JSON mode | [Image input](/oss/langchain/messages#multimodal) | Audio input | Video input | [Token-level streaming](/oss/langchain/streaming#llm-tokens) | Native async | [Token usage](/oss/langchain/models#token-usage) | [Logprobs](/oss/langchain/models#log-probabilities) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |


Integration package connecting Sarvam AI chat completions with LangChain.

## Installation


with `uv` inside the package:

```bash
uv add langchain-sarvam
```

## Setup

```python
# Set the SARVAM API key
sarvam_Api_key = os.getenv("SARVAM_API_KEY")
```

## Usage
### Basic Usage

```python
from langchain_sarvam import ChatSarvam

llm = ChatSarvam(model="sarvam-m", temperature=0.2, max_tokens=128)
resp = llm.invoke([("system", "You are helpful"), ("human", "Hello!")])
print(resp.content)
```


### Batch Processing

```python
from langchain_sarvam import ChatSarvam
from langchain_core.messages import HumanMessage

chat = ChatSarvam(model="sarvam-m")

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

chat = ChatSarvam(model="sarvam-m")

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
for chunk in ChatSarvam(model="sarvam-m", streaming=True).stream("Tell me a joke"):
    print(chunk.text, end="")
```
 
