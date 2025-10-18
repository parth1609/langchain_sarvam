# langchain-sarvam

Integration package connecting Sarvam AI chat completions with LangChain.

## Installation


with `uv` inside the package:

```bash
uv pip install langchain-sarvam
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

### Language-Specific Usage

```python
from langchain_sarvam import ChatSarvam

llm = ChatSarvam(
    model="sarvam-m",
    temperature=0.7,
    sarvam_api_key=os.getenv("SARVAM_API_KEY")
)

response = llm.invoke([
    ("system", "talk in Hindi"),
    ("human", "what is color of sky?"),
])
print(response.content)  # Output: आसमान का रंग नीला होता है...
```

### Advanced Content Generation

```python
from langchain_sarvam import ChatSarvam

llm = ChatSarvam(model="sarvam-m")

# Generate blog post outline
response = llm.invoke("create the outline for the blog post outline for blog topic - AI engineering.")
print(response.content)
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
 