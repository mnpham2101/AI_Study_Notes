# Working with Llama 3

## Introduction

Llama 3 is an open-weight large language model (LLM). You can download it and run it on your own machine.

Why run it locally:

- **Privacy and security**: prompts and data never leave your machine.
- **Cost**: no per-token API fees.
- **Customization**: you control the model, its settings, and its version.

This guide uses the `llama-cpp-python` library to run Llama 3 in Python. It covers:

- Download and installation.
- Importing the model and sending a basic prompt.
- Use cases: tuning parameters, zero/one/multi-shot prompts, JSON output, and conversations.

"Llama 3" is the model family. Meta also released later versions (3.1, 3.2, 3.3). The code here works with any chat model in GGUF format.

References:

- Meta Llama: <https://www.llama.com/>
- Llama 3 announcement: <https://ai.meta.com/blog/meta-llama-3/>
- `llama-cpp-python` docs: <https://llama-cpp-python.readthedocs.io/>

---

## Credits

| Project | Developer | Role |
|---|---|---|
| **Llama 3** (model) | **Meta AI** (Meta Platforms, Inc.) | Trained and released the model weights (April 2024). Sizes include 8B and 70B parameters. |
| **llama.cpp** | **Georgi Gerganov** and open-source contributors | C/C++ engine that runs LLMs efficiently on CPUs and GPUs. Created the GGUF format. |
| **llama-cpp-python** | **Andrei Betlen** and contributors | Python bindings for llama.cpp. |
| **GGUF conversions** | Community publishers on Hugging Face (for example Unsloth, bartowski, QuantFactory) | Converted and quantized Llama 3 weights into GGUF files. |
| **Hugging Face** | Hugging Face, Inc. | Hosts the model files. |

Licensing:

- Llama 3 is released under the **Meta Llama 3 Community License**. It allows commercial use with conditions (for example attribution and an acceptable-use policy).
- Read the license before you use or redistribute the model.
- License text: <https://www.llama.com/llama3/license/>

Course note: the structure and several examples in this guide follow the "Working with Llama 3" course material (instructor: Imtihan Ahmed, DataCamp).

---

## 1. Download and Installation

**Key concepts**

- **Weights**: the trained parameters of the model. They are large files.
- **GGUF**: a single-file model format used by llama.cpp. It stores weights and metadata.
- **Quantization**: storing weights with fewer bits to save memory. For example, `Q4_K_M` uses about 4 bits per weight. It is smaller and faster, with a small loss in quality.
- **Instruct model**: a Llama 3 variant tuned to follow instructions and chat. Use it for the examples below.

Docs: [llama-cpp-python – installation](https://llama-cpp-python.readthedocs.io/en/latest/#installation)

### 1.1 Requirements

- Python 3.8 or newer.
- A C/C++ compiler and CMake, if pip has to build the library (Windows: Visual Studio Build Tools).
- Memory: an 8B model at `Q4_K_M` needs about 5–6 GB of RAM. A GPU is optional.
- Disk space: about 5 GB for the 8B model at `Q4_K_M`.

### 1.2 Install the library

```bash
pip install llama-cpp-python
```

If the build fails, install a prebuilt CPU wheel:

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

For GPU support, see the installation page for CUDA, Metal, and other backends.

### 1.3 Download the model

Download a GGUF file from Hugging Face.

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="bartowski/Meta-Llama-3-8B-Instruct-GGUF",
    filename="Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
)
print(model_path)  # local path to the .gguf file
```

Notes:

- Repository and file names change. Check the repository's file list for the exact name.
- Meta's own repositories are "gated". You must accept the license on Hugging Face and log in with a token. Community GGUF repositories are usually not gated.
- Alternatively, download the `.gguf` file in your browser and keep the path.

---

## 2. Importing the Model and Basic Prompts

**Key concepts**

- `Llama` is the main class. It loads a GGUF model into memory.
- `model_path` points to the `.gguf` file.
- Calling the object, `llm(prompt)`, runs a **text completion**. The model continues the text you give it.
- The result is a Python dictionary.

Docs: [llama-cpp-python – high-level API](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/)

### 2.1 Load the model

```python
from llama_cpp import Llama

llm = Llama(
    model_path="path/to/model.gguf",
    n_ctx=2048,      # context window in tokens (prompt + reply)
    verbose=False,   # hide the loading logs
)
```

### 2.2 Send a basic prompt

```python
output = llm("What are some ways to improve customer retention?")
```

The flow: the model receives the prompt, processes it, and returns a response.

### 2.3 Unpack the output

The output is a dictionary in an OpenAI-like format.

```python
print(output)
# {'id': 'cmpl-...', 'object': 'text_completion', 'created': 1715222298,
#  'model': 'path/to/model.gguf',
#  'choices': [{'text': 'Improving customer retention is crucial...', ...}], ...}
```

Read the generated text from the first item in `choices`:

```python
print(output["choices"][0]["text"])
```

Also check how many tokens were used:

```python
print(output["usage"])  # prompt_tokens, completion_tokens, total_tokens
```

---

## 3. Use Cases

### 3.1 Tuning parameters

**Key concepts**

- **Decoding parameters** control how the model picks the next token. They change tone and variety without changing the model.
- Lower values give predictable, factual text. Higher values give varied, creative text.

| Parameter | What it does | Low value | High value |
|---|---|---|---|
| `temperature` | Controls randomness. Usually 0 to 1. | Predictable | Creative |
| `top_k` | Picks only from the *k* most likely tokens. | `1`: always the top token | `50`: more diverse |
| `top_p` | Picks from the smallest set of tokens whose probabilities add up to *p* (nucleus sampling). | Near `0`: little variation | Near `1`: varied |
| `max_tokens` | Limits the reply length, in tokens. | Short, may cut off | Long |

Docs: [llama-cpp-python – `__call__`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__call__)

**Concise, factual output**

```python
output_concise = llm(
    "Describe an electric car.",
    temperature=0.2,
    top_k=1,
    top_p=0.4,
    max_tokens=20,
)
print(output_concise["choices"][0]["text"])
```

**Creative output**

```python
output_creative = llm(
    "Describe an electric car.",
    temperature=0.8,
    top_k=10,
    top_p=0.9,
    max_tokens=100,
)
print(output_creative["choices"][0]["text"])
```

Tips:

- Change one parameter at a time to see its effect.
- Check `choices[0]["finish_reason"]`. The value `"length"` means `max_tokens` cut the reply off.
- Use a low temperature for extraction and classification. Use a higher one for brainstorming and marketing copy.

### 3.2 Zero-shot, one-shot, and multi-shot prompts

**Key concepts**

- **Shot**: a worked example included in the prompt.
- **Zero-shot**: no examples. The model relies on the instruction alone.
- **One-shot**: one example.
- **Multi-shot (few-shot)**: several examples. They show the format and the task more clearly.
- More examples usually improve accuracy and format consistency. They also use more tokens.

**Zero-shot**

```python
prompt = "Classify the sentiment as positive or negative: 'The battery dies too fast.'"
out = llm(prompt, temperature=0, max_tokens=5)
print(out["choices"][0]["text"])
```

**One-shot**

```python
prompt = """Classify the sentiment as positive or negative.

Review: 'I love the screen.'
Sentiment: positive

Review: 'The battery dies too fast.'
Sentiment:"""
out = llm(prompt, temperature=0, max_tokens=5)
```

**Multi-shot**

```python
prompt = """Classify the sentiment as positive or negative.

Review: 'I love the screen.'
Sentiment: positive

Review: 'Support never answered my emails.'
Sentiment: negative

Review: 'Fast shipping and great quality.'
Sentiment: positive

Review: 'The battery dies too fast.'
Sentiment:"""
out = llm(prompt, temperature=0, max_tokens=5)
print(out["choices"][0]["text"].strip())
```

Tips:

- Keep the example format identical to the question format.
- End the prompt where you want the model to start writing.
- Use `max_tokens` and `stop=["\n"]` to end the reply after the label.

### 3.3 Getting JSON structured responses

**Key concepts**

- **Structured output**: a reply your code can parse, such as JSON.
- Use `create_chat_completion()` with `response_format={"type": "json_object"}`. The library then constrains the output to valid JSON.
- Still describe the fields you want in the prompt.
- The reply is a string in `message["content"]`. Parse it with `json.loads()`.

Docs: [llama-cpp-python – JSON and JSON schema mode](https://llama-cpp-python.readthedocs.io/en/latest/#json-and-json-schema-mode)

```python
import json

response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You reply only with JSON."},
        {"role": "user", "content": "List two EV brands with keys 'brand' and 'country'."},
    ],
    response_format={"type": "json_object"},
    temperature=0.2,
)

data = json.loads(response["choices"][0]["message"]["content"])
print(data)
```

Constrain the structure with a schema:

```python
response = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Describe the Tesla Model 3."}],
    response_format={
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "range_km": {"type": "number"},
            },
            "required": ["name", "range_km"],
        },
    },
)
```

Notes:

- Use `json.loads()` inside `try`/`except json.JSONDecodeError`. A reply cut off by `max_tokens` is not valid JSON.
- Leave enough `max_tokens` for the whole JSON object.

### 3.4 Building a conversation

**Key concepts**

- **Chat roles** guide the model's replies:
  - `system`: sets personality, style, and rules.
  - `user`: the person asking the question.
  - `assistant`: the model's earlier replies.
- `create_chat_completion(messages=[...])` accepts a list of role-tagged messages.
- The model has **no memory** between calls. To continue a chat, resend the whole history, including its own earlier replies.
- The reply is at `response["choices"][0]["message"]["content"]`.

Docs: [llama-cpp-python – chat completion](https://llama-cpp-python.readthedocs.io/en/latest/#chat-completion)

**A basic chat call**

```python
system_message = "You are a business consultant who gives data-driven answers."
user_message = "What are the key factors in a successful marketing strategy?"

message_list = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_message},
]

response = llm.create_chat_completion(messages=message_list)
print(response["choices"][0]["message"]["content"])
```

**Building up the history**

After each reply, append it to the list as an `assistant` message. Then append the next `user` message and call the model again. The list grows every turn.

```python
messages = [{"role": "system", "content": "You are a data science tutor who gives short, simple explanations."}]

user_qs = ["Why is Python so popular?", "Summarize that in one sentence."]

for q in user_qs:
    messages.append({"role": "user", "content": q})                  # 1. add the question

    response = llm.create_chat_completion(messages=messages)         # 2. send the full history
    reply = response["choices"][0]["message"]["content"]

    messages.append({"role": "assistant", "content": reply})         # 3. add the reply

    print("User:     ", q)
    print("Assistant:", reply)
    print("History size:", len(messages), "\n")
```

How the history changes:

| After | `messages` holds | Size |
|---|---|---|
| Start | `system` | 1 |
| Turn 1 | `system`, `user`, `assistant` | 3 |
| Turn 2 | `system`, `user`, `assistant`, `user`, `assistant` | 5 |

- In turn 2, the model sees the first question and answer. This is why "Summarize that" works.
- Without step 3, the model forgets its own earlier replies.
- The whole list is resent every turn, so the prompt grows and `n_ctx` can fill up.

#### a) Creating a conversation class

This builds on the loop above. The loop kept `messages` in a variable and repeated three steps by hand: add the question, call `create_chat_completion()`, add the reply. The class packages those steps into one method, so you no longer repeat them.

- The history lives inside the object (`self.history`) instead of a loose variable.
- `create_completion()` is the loop body: it calls `create_chat_completion()` from section 3.4 on `self.history`.
- `__init__` stores the model and starts `history` with the `system` message.
- `create_completion` appends the user input, calls the model, appends the model's reply, and returns the reply text.
- The reply message returned by the model already has `role` and `content`, so it can be appended as is.

```python
class Conversation:
    def __init__(self, llm: Llama, system_prompt='', history=[]):
        self.llm = llm
        self.system_prompt = system_prompt
        self.history = [{"role": "system", "content": self.system_prompt}] + history

    def create_completion(self, user_prompt=''):
        self.history.append({"role": "user", "content": user_prompt})  # Append input
        output = self.llm.create_chat_completion(messages=self.history)
        conversation_result = output['choices'][0]['message']
        self.history.append(conversation_result)  # Append output
        return conversation_result['content']  # Return output
```

Note: `[{...}] + history` builds a new list, so the mutable default `history=[]` is never modified. Passing `history=None` and handling it in the constructor is a safer habit.

#### b) Single-turn conversation

One question, one answer. Create an object from the class in (a), then call its `create_completion()` method once. The object starts with only the `system` message, so there are no earlier turns to remember.

```python
conv = Conversation(llm, system_prompt="You are a customer support agent. Be brief and polite.")

print(conv.create_completion("How do I reset my password?"))
```

#### c) Multi-turn conversation

This reuses the same class and method from (a) and (b). The only change is that you call `create_completion()` more than once on the **same** `conv` object.

- `conv` is an instance of `Conversation`. It keeps its own `history`.
- Each call to `create_completion()` appends the user prompt and the model's reply to that history, as in the loop above.
- The next call then sends the whole history to `create_chat_completion()`, so the model sees the earlier turns. That is why a follow-up like "Summarize that" works.

```python
conv = Conversation(llm, system_prompt="You are a data science tutor who gives short, simple explanations.")

print(conv.create_completion("Why is Python so popular?"))
print(conv.create_completion("Summarize that in one sentence."))  # "that" refers to the last answer
print(len(conv.history))  # 5: system + 2 user + 2 assistant
```

Keep long chats under control:

- The history grows with every turn. It must fit in `n_ctx` together with the reply.
- Drop or summarize old turns, but keep the `system` message.

```python
def trim(history, keep_last=6):
    return history[:1] + history[1:][-keep_last:]

conv.history = trim(conv.history)
```

To start a new conversation, create a new `Conversation` object.

---

## Summary

| Task | How |
|---|---|
| Install | `pip install llama-cpp-python` |
| Get a model | Download a `.gguf` file from Hugging Face |
| Load | `Llama(model_path="...", n_ctx=2048)` |
| Basic prompt | `llm("prompt")` then `output["choices"][0]["text"]` |
| Tune output | `temperature`, `top_k`, `top_p`, `max_tokens` |
| Few-shot | Put examples in the prompt |
| JSON | `create_chat_completion(..., response_format={"type": "json_object"})` |
| Conversation | Resend `system`, `user`, and `assistant` messages each turn |
