# Working with Hugging Face

## Introduction

Hugging Face is "the home of the AI community". It is a platform and a set of open-source libraries for finding, sharing, and running machine learning models and datasets.

This guide covers:

- What the Hugging Face Hub offers: models, datasets, and related concepts.
- Downloading a dataset and running a first pipeline.
- Using pipelines for text generation, summarization, and text classification.
- Using Auto classes for more control, with a document Q&A example.

Model names and outputs shown below are illustrative. Models on the Hub change or disappear, and scores vary by library version, so check the model card.

References:

- Hub docs: <https://huggingface.co/docs/hub/index>
- Transformers: <https://github.com/huggingface/transformers>
- Datasets: <https://huggingface.co/docs/datasets>
- Inference Providers: <https://huggingface.co/docs/inference-providers/en/index>

---

## 1. Concepts and Features

### 1.1 Installing the libraries

```bash
pip install transformers datasets huggingface_hub pypdf torch
```

### 1.2 The Hub, models, and model cards

**Key concepts**

- **Hub**: the website <https://huggingface.co> that hosts models, datasets, and demo apps (Spaces). Each item lives in a git-based repository.
- **Model**: a set of trained weights plus configuration, built for a task such as text generation or classification. Models are named `owner/name`, for example `openai-community/gpt2`.
- **Model card**: the README of a model repo. It states the intended task, training data, limits, and example code. Read it before you use a model.
- **Task**: the kind of job a model does, such as `text-generation`, `summarization`, `text-classification`, or `question-answering`. You can filter models by task on the Hub.
- **Pre-trained model**: a model already trained by someone else. You download and use it, or fine-tune it for your data.

Docs: [Hub documentation](https://huggingface.co/docs/hub/index) · [Models](https://huggingface.co/models)

### 1.3 Datasets

**Key concepts**

- **Dataset**: a collection of examples (rows) with named features (columns), hosted on the Hub. Examples are text, audio, images, or tables.
- **`datasets` library**: downloads a dataset, caches it, and lets you filter and slice it.
- **Split**: a named part of a dataset, usually `train`, `validation`, and `test`.
- **Apache Arrow**: the columnar memory format the library uses. It is fast and memory-mapped, so a large dataset does not have to fit in RAM.
- The library works with the usual data formats too, such as CSV, JSON, and Parquet.

Docs: [Datasets on the Hub](https://huggingface.co/datasets) · [Loading a dataset](https://huggingface.co/docs/datasets/loading) · [Apache Arrow](https://arrow.apache.org/overview/)

### 1.4 The Transformers library, pipelines, and Auto classes

**Key concepts**

- **`transformers` library**: simplifies working with pre-trained models. It handles downloading, loading, and running them.
- **Pipeline**: a high-level function, `pipeline(task, model)`. It bundles a tokenizer, a model, and post-processing into one callable. It is the fast and simple option.
- **Tokenizer**: converts text into tokens (IDs) the model understands, and back. Each model needs the tokenizer it was trained with.
- **Auto classes**: `AutoModel...` and `AutoTokenizer`. They pick the right class from a model name, so you load a model and tokenizer yourself. This is the flexible option with more control.
- Rule of thumb: pipelines are quick, Auto classes are flexible.

Docs: [Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines) · [Auto classes](https://huggingface.co/docs/transformers/model_doc/auto)

### 1.5 Inference: local vs. providers

**Key concepts**

- **Inference**: using a trained model to produce output for new input.
- **Local inference**: the model downloads and runs on your own machine.
  - Free and convenient.
  - Slow and resource-intensive for large models (CPU, GPU, RAM).
- **Inference providers**: hosted services (such as Together) that run the model for you. You call them through one API.
  - Fast.
  - Free to get started, with usage limits. Heavier use is billed.
  - Needs a Hugging Face access token.

Docs: [Inference Providers](https://huggingface.co/docs/inference-providers/en/index)

Run a hosted model with `InferenceClient`:

```python
import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="together",
    api_key=os.environ["HF_TOKEN"],  # your Hugging Face access token
)

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)

print(completion.choices[0].message)
# The capital of France is **Paris**. It is known for its iconic landmarks ...
```

The chat interface matches the OpenAI Chat Completions style, so `messages` and roles work the same way.

Set the token in your shell and never commit it:

```bash
# macOS / Linux
export HF_TOKEN="hf_..."

# Windows PowerShell
$env:HF_TOKEN = "hf_..."
```

---

## 2. Datasets and a First Pipeline

### 2.1 Download and install a dataset

**Key concepts**

- `pip install datasets` installs the package.
- With it you can access, download, use, and share datasets.
- `load_dataset("owner/name")` downloads a dataset and returns a `DatasetDict` (all splits). The data is cached locally.
- The `split` parameter returns one split as a `Dataset`.

Docs: [Loading a dataset](https://huggingface.co/docs/datasets/loading)

```bash
pip install datasets
```

```python
from datasets import load_dataset

# All splits
data = load_dataset("IVN-RIN/BioBERT_Italian")

# One split only
data = load_dataset("IVN-RIN/BioBERT_Italian", split="train")
```

Manipulate the data:

```python
data = load_dataset("IVN-RIN/BioBERT_Italian", split="train")

# Filter rows that contain the word " bella "
filtered = data.filter(lambda row: " bella " in row["text"])
print(filtered)
# Dataset({ features: ['text'], num_rows: 1122 })

# Select the first two rows
sliced = filtered.select(range(2))
print(sliced)
# Dataset({ features: ['text'], num_rows: 2 })

# Read the 'text' feature of the first row
print(sliced[0]["text"])
```

Docs: [Process – select and filter](https://huggingface.co/docs/datasets/process#select-and-filter)

### 2.2 Run a pipeline

**Key concepts**

- A **pipeline instance** is the object `pipeline(...)` returns. Create it once, then call it like a function.
- `task` tells it what job to do. `model` names the Hub model to use.
- The first call downloads the model. Later calls reuse the cached copy.
- Without `model`, the pipeline uses a default model for the task. Always name the model so results are reproducible.

Docs: [Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)

```python
from transformers import pipeline

# Create the pipeline instance
gpt2_pipeline = pipeline(task="text-generation", model="openai-community/gpt2")

# Call it with a prompt
print(gpt2_pipeline("What if AI"))
# [{'generated_text': 'What if AI wouldn't be used? ...'}]
```

The result is a list of dicts. The text is under `generated_text`. Output differs on each run because generation is random.

Model card: <https://huggingface.co/openai-community/gpt2>

---

## 3. Performing Tasks with a Pipeline

Every task follows the same pattern:

1. Create the pipeline with `task` and `model`.
2. Call it with your text.
3. Read the result. Its shape depends on the task.

### 3.1 Text generation

**Key concepts**

- **Text generation**: the model continues a prompt, one token at a time.
- The `gpt2` model is a base model that continues text. It does not follow instructions or answer questions.
- `max_new_tokens`: the maximum number of tokens to generate, not counting the prompt. Prefer it over `max_length`, which includes the prompt.
- `num_return_sequences`: how many different continuations to return.
- `do_sample=True`: sample from the model's probabilities, which gives varied text. Without it, decoding is greedy. Several return sequences need sampling (or beam search).
- `temperature`: higher values give more random text. Use it together with `do_sample=True`.

Docs: [Text generation](https://huggingface.co/docs/transformers/main_classes/text_generation) · [Task page](https://huggingface.co/tasks/text-generation)

```python
from transformers import pipeline

gpt2_pipeline = pipeline(task="text-generation", model="openai-community/gpt2")

results = gpt2_pipeline(
    "What if AI",
    max_new_tokens=10,
    num_return_sequences=2,
    do_sample=True,
)

for result in results:
    print(result["generated_text"])
# What if AI had never existed?
# What if AI could be really smarter than us?
```

### 3.2 Extractive text summarization

**Key concepts**

- **Summarization**: shorten a long text and keep the key points.
- **Extractive** summarization selects the key sentences from the original text, unchanged.
  - Efficient and needs fewer resources.
  - Lacks flexibility. The result may be less cohesive.
  - Use cases: legal documents (highlight key clauses), financial research (extract insights).
- **Abstractive** summarization generates new, rephrased text, as a person would write a summary.
  - Clearer and more readable.
  - Needs more resources and processing. It can also state something the source never said.
  - Use cases: news articles (concise summaries), content recommendations (compelling descriptions).
- `min_new_tokens` and `max_new_tokens` set the lower and upper bound of the summary length, in generated tokens.

Docs: [Summarization task](https://huggingface.co/tasks/summarization) · [Summarization guide](https://huggingface.co/docs/transformers/tasks/summarization)

| | Extractive | Abstractive |
|---|---|---|
| Output text | Copied from the source | Newly generated |
| Readability | Can feel choppy | Fluent |
| Cost | Low | Higher |
| Risk | Misses context between sentences | Can invent facts |

Extractive example, with a length range:

```python
from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="nyamuda/extractive-summarization",
)

text = "This is my really large text about Data Science..."

summary_text = summarizer(
    text,
    min_new_tokens=10,    # at least 10 tokens
    max_new_tokens=150,   # at most 150 tokens
)

print(summary_text[0]["summary_text"])
# "data science is a field that combines mathematics, statistics ..."
```

Notes:

- The result is a list. The summary is in `[0]["summary_text"]`.
- The length limits can also be passed when you create the pipeline, as the course slide does: `pipeline(task="summarization", min_new_tokens=10, max_new_tokens=150)`.
- `min_new_tokens` must be smaller than `max_new_tokens`.

For contrast, an abstractive model uses the same code. Only the model changes:

```python
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
print(summarizer(text)[0]["summary_text"])
# "The global data science platform market is projected to reach $140.9 billion by 2025 ..."
```

The sentences are newly written, not copied from the text.

### 3.3 Text classification

**Key concepts**

- **Text classification**: assign a label to a text. The pipeline returns the label and a `score` (confidence between 0 and 1).
- Use `task="text-classification"`. The task changes what the model does through the `model` you pick:
  - Sentiment analysis: label text by its emotional tone (reviews, social media).
  - Grammatical correctness: check if the grammar is right (grammar checkers, language learning).
  - QNLI: check if a premise answers a question (Q&A systems, fact-checking).
  - Dynamic category assignment: assign categories you choose at run time (content moderation, recommendation).
- Labels such as `LABEL_0` and `LABEL_1` are generic. Read the model card to see what each means.

Docs: [Text classification task](https://huggingface.co/tasks/text-classification) · [Zero-shot classification](https://huggingface.co/tasks/zero-shot-classification)

Sentiment analysis, for reference:

```python
from transformers import pipeline

my_pipeline = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

print(my_pipeline("Wi-Fi is slower than a snail today!"))
# [{'label': 'NEGATIVE', 'score': 0.99}]
```

#### 3.3.1 Grammar checker

Checks whether a sentence is grammatically correct.

```python
grammar_checker = pipeline(
    task="text-classification",
    model="abdulmatinomotoso/English_Grammar_Checker",
)

print(grammar_checker("He eat pizza every day."))
# [{'label': 'LABEL_0', 'score': 0.99}]
```

`LABEL_0` marks the sentence as **not acceptable**, which is right here ("He eats" is correct). Check the model card for the label mapping. Try a correct sentence and compare the labels.

#### 3.3.2 QNLI (question natural language inference)

QNLI decides whether a sentence contains the answer to a question. Pass the question and the sentence as one string, separated by a comma.

```python
classifier = pipeline(
    task="text-classification",
    model="cross-encoder/qnli-electra-base",
)

print(classifier("Where is Seattle located?, Seattle is located in Washington state."))
# [{'label': 'LABEL_0', 'score': 0.997}]
```

Notes:

- The model sees the pair of texts. Its label says whether the sentence answers the question.
- For this cross-encoder, a higher score means the sentence answers the question. The sample output shows `LABEL_0` with `0.997`, so check the model card for the meaning of the label before relying on it.
- A pair can also be passed as a dict: `classifier({"text": question, "text_pair": sentence})`.

#### 3.3.3 Dynamic category assignment (zero-shot classification)

**Key concepts**

- **Zero-shot classification**: classify text into categories the model was never trained on. You supply the candidate labels at run time.
- No training or fine-tuning is needed. Change the label list and the same model works.
- The model scores each label against the text. It uses a natural language inference (NLI) model such as `facebook/bart-large-mnli`.
- The output has `labels` and `scores`, sorted from the best match down.

```python
classifier = pipeline(
    task="zero-shot-classification",
    model="facebook/bart-large-mnli",
)

text = "Hey, DataCamp; we would like to feature your courses in our newsletter!"
categories = ["marketing", "sales", "support"]

output = classifier(text, categories)

print(f"Top Label: {output['labels'][0]} with score: {output['scores'][0]}")
# Top Label: support with score: 0.8183
```

Notes:

- The course sample reports `support`, though this message reads like `marketing`. Zero-shot models can be wrong, so test them on your own examples.
- Descriptive label names help. Phrases such as "newsletter feature request" often work better than one bare word.
- Set `multi_label=True` if a text can belong to several categories. Scores then no longer sum to 1.

**Challenges of text classification**

- Ambiguous or sarcastic text.
- Domain-specific language that the model did not see in training.
- Imbalanced classes and biased training data.
- Confidence scores are not always calibrated. A score of 0.99 can still be wrong.

---

## 4. Auto Classes

**Key concepts**

- A pipeline picks and wires everything for you. Auto classes let you do it yourself.
- Use them for more control and customization:
  - **Text preprocessing**: clean and tokenize for a specific use case.
  - **Thresholding**: prioritize key categories in classification tasks.
  - **Complex workflows**: control multi-stage processing and integration.
- `AutoModelFor<Task>`: loads a model with the right head for a task.
- `AutoTokenizer`: loads the tokenizer paired with the model. Always use the tokenizer that matches the model, because different models split text differently.
- `from_pretrained("name")` downloads and loads from the Hub.

Docs: [Auto classes](https://huggingface.co/docs/transformers/model_doc/auto) · [Tokenizers](https://huggingface.co/docs/transformers/main_classes/tokenizer)

Different models split text in different ways:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print(tokenizer.tokenize("AI: Helping robots think and humans overthink :)"))
# ['ai', ':', 'helping', 'robots', 'think', 'and', 'humans', 'over', '##thi', '##nk', ':', ')']
```

`bert-base-cased` keeps the case and splits words differently (`'Help', '##ing'`). The `##` prefix marks a token that continues the previous word.

Build a pipeline from Auto classes:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

name = "distilbert-base-uncased-finetuned-sst-2-english"

my_model = AutoModelForSequenceClassification.from_pretrained(name)
my_tokenizer = AutoTokenizer.from_pretrained(name)

my_pipeline = pipeline(
    task="sentiment-analysis",
    model=my_model,
    tokenizer=my_tokenizer,
)
```

### 4.1 Document Q&A

**Key concepts**

- **Document question answering**: answer a question from the content of a document. It needs a document (the **context**) and a question.
- The model used here is **extractive**. It finds the span of the context that answers the question, so the answer is text copied from the document.
- Use cases: legal (find contract clauses), finance (extract key figures), support (retrieve answers from manuals).
- Example scenario: automate HR queries over `US-Employee_Policy.pdf`, which saves HR time.
- Steps:
  1. Read the PDF and extract its text.
  2. Create a `question-answering` pipeline.
  3. Pass a `question` and the `context`.
  4. Wrap the steps in functions to automate queries.

Docs: [Question answering task](https://huggingface.co/tasks/question-answering) · [Question answering guide](https://huggingface.co/docs/transformers/tasks/question_answering) · [pypdf](https://pypdf.readthedocs.io/)

**Step 1: Extract text from a PDF with `pypdf`**

```python
from pypdf import PdfReader

reader = PdfReader("US-Employee_Policy.pdf")

# Extract text from all pages
document_text = ""
for page in reader.pages:
    document_text += page.extract_text()

print(document_text[:60])
# Welcome to the US Employee Policy document...
```

`extract_text()` works on text-based PDFs. A scanned PDF needs OCR first.

**Step 2: Q&A pipeline**

```python
from transformers import pipeline

qa_pipeline = pipeline(
    task="question-answering",
    model="distilbert-base-cased-distilled-squad",
)

question = "How many volunteer days are offered annually?"

result = qa_pipeline(question=question, context=document_text)
print(f"Answer: {result['answer']}")
# Answer: 1
```

The result is a dict with `answer`, `score`, and the character positions `start` and `end` of the answer in the context. A low `score` means the model is unsure.

**Step 3: Build the same pipeline from Auto classes**

This version gives the model and tokenizer explicitly, so you can swap or configure either.

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

name = "distilbert-base-cased-distilled-squad"

qa_model = AutoModelForQuestionAnswering.from_pretrained(name)
qa_tokenizer = AutoTokenizer.from_pretrained(name)

qa_pipeline = pipeline(
    task="question-answering",
    model=qa_model,
    tokenizer=qa_tokenizer,
)

print(qa_pipeline(question=question, context=document_text)["answer"])
```

A pipeline splits a long context into overlapping chunks and returns the best answer.

**Step 4: Without a pipeline, using the model directly**

This shows what the pipeline does inside. The model scores every token as the possible start and end of the answer.

```python
import torch

inputs = qa_tokenizer(
    question,
    document_text,
    return_tensors="pt",
    truncation="only_second",  # cut the document, never the question
    max_length=384,
)

with torch.no_grad():
    outputs = qa_model(**inputs)

start = outputs.start_logits.argmax()
end = outputs.end_logits.argmax()

answer_ids = inputs["input_ids"][0][start : end + 1]
print(qa_tokenizer.decode(answer_ids, skip_special_tokens=True))
```

Limits: this shortcut only reads the first 384 tokens and ignores the rest of the document, and it does not check that `end >= start`. Use the pipeline for real documents.

**Step 5: Wrap it in functions**

```python
from pypdf import PdfReader
from transformers import pipeline

qa_pipeline = pipeline(
    task="question-answering",
    model="distilbert-base-cased-distilled-squad",
)

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "".join(page.extract_text() or "" for page in reader.pages)

def ask_document(question: str, context: str) -> str:
    result = qa_pipeline(question=question, context=context)
    return result["answer"]

policy = read_pdf("US-Employee_Policy.pdf")
print(ask_document("How many volunteer days are offered annually?", policy))
print(ask_document("How many sick days do employees get?", policy))
```

Read the PDF once and reuse the text for many questions.

Notes:

- An extractive model returns a span of the document. It cannot write a new answer, and it always returns something, even if the document has no answer. Check `score`.
- To get an answer written in full sentences, use a generative model (see 1.5 and 3.1).

---

## Summary

| Task | Code |
|---|---|
| Install | `pip install transformers datasets huggingface_hub` |
| Load a dataset | `load_dataset("owner/name", split="train")` |
| Filter / slice a dataset | `data.filter(fn)`, `data.select(range(n))` |
| Run a model locally | `pipeline(task=..., model=...)` |
| Run a hosted model | `InferenceClient(provider=..., api_key=...)` |
| Generate text | `task="text-generation"`, `max_new_tokens`, `num_return_sequences` |
| Summarize | `task="summarization"`, `min_new_tokens`, `max_new_tokens` |
| Classify text | `task="text-classification"` |
| Classify with your own labels | `task="zero-shot-classification"`, `classifier(text, categories)` |
| Answer from a document | `task="question-answering"`, `question=`, `context=` |
| Full control | `AutoModelFor<Task>`, `AutoTokenizer`, `.from_pretrained()` |
