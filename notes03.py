#### Testing LLMs
## Simple Code using Ollama for Sentiment Analysis
import ollama

def analyse_sentiment(text):
    prompt = f"Analyse the sentiment of this text and give me POSITIVE, NEGATIVE and NEUTRAL for this {text}"
    response = ollama.chat(model='qwen2.5:latest',messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

texts = [
    "I love this product",
    "I hate the food in this resturant",
    "The movie was just okay"
]

for text in texts:
    sentiment = analyse_sentiment(text)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}")
"""
Text: I love this product
Sentiment: The sentiment of the text "I love this product" is POSITIVE.
- **POSITIVE**: The phrase "love" indicates a strong positive feeling towards the product.
"""

## Text Summarization (Classification) 📚
def summarize_text(text):
    prompt = f"Please provide a consice summary of the given {text} and keep it as simple as possible"
    response = ollama.chat(model='llama3.2:latest',messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

text = """{ paragraphs can be added here}"""
summary = summarize_text(text)
print(f"Summary: {summary}")

## NLP Library (Transformer) from HuggingFace
"""
* Hugging Face is an open-source AI platform that provides pre-trained models, datasets, and tools to build, train, 
and deploy NLP, vision, and multimodal AI models.
* It hosts a vast collection of models and datasets contributed by the AI community.
* The `transformers` library by Hugging Face offers easy access to state-of-the-art pre-trained models for various 
NLP tasks like text classification, translation, summarization, question answering, and more.
====================================================================================================================
* Hugging Face Pipeline is a high-level wrapper that combines model loading, tokenization, preprocessing, inference, 
and postprocessing into **one simple function call**.
* It **automatically downloads, caches, and loads** the model and tokenizer from the Hugging Face Hub.
* Hugging Face pipelines abstract the entire ML inference workflow into a single, easy-to-use API.
====================================================================================================================
#What a pipeline does (flow):
* Selects a task (e.g., sentiment-analysis, translation)
* Loads model + tokenizer 
    - Tokenizer converts raw text into numerical tokens that a model can understand and converts model outputs back to text.
* Preprocesses input into model-ready format
* Runs inference # feeds input through the model to get raw predictions
* Postprocesses output into human-readable results
* Returns clean output (e.g., `{'label': 'POSITIVE', 'score': 0.95}`)
====================================================================================================================
# text-classification / sentiment-analysis
* Used for text category prediction (commonly sentiment).
* Defaults to 'distilbert-base-uncased-finetuned-sst-2-english' for sentiment analysis.
=====================================================================================================================
#device parameter
* `device = -1` → Run on CPU
* `device = 0` → First GPU
* Use `-1` when PyTorch is CPU-only to avoid CUDA errors.
=====================================================================================================================
# Common Supported Pipeline Tasks
* sentiment-analysis → Positive/Negative classification
* text-classification → Multi-class classification
* translation → Language translation
* summarization → Text summarization
* question-answering → Answer extraction from context
* ner → Named Entity Recognition
* image-classification → Image categorization

"""
# !pip install transformers
# !pip install evaluate
# !pip install tensorflow
# !pip install tf-keras
# !pip install scikit-learn
# !pip install torch # !pip install torch --index-url https://download.pytorch.org/whl/cu118
from transformers import pipeline
classifier = pipeline("text-classification", device=-1) # text-classification and sentiment-analysis are same. 
classifier("I think I love this food eventhough many people says its terrible and bad and worst")

## Text Summarization Classification
summarizer = pipeline("summarization", device=-1)
summary = summarizer(text)
print(summary[0]['summary_text'])

## Testing custom LLM model from HF 🧠🤖
classifier = pipeline("text-classification", model="ExecuteAutomation/bert-base-text-classification-model")

## Zero Shot Classification
"""
* Zero-Shot Classification is a technique where an LLM/classifier categorizes text into labels it was not explicitly trained on, 
using natural language understanding.
* Instead, the model uses its general language understanding to decide which label best fits the input.
* “Zero-shot means no model training at all”
    ✅ NO: Model is pre-trained, but not fine-tuned for this task.
* Example
    Text: “I want a refund”
    Labels: ["billing", "technical", "sales"]
        → Output: billing
* Zero-shot → Model predicts without seeing examples.
* Few-shot → Model learns from examples in the prompt.
* Default model for zero-shot-classification is `facebook/bart-large-mnli`
"""
classifier = pipeline("zero-shot-classification")
classifier("I love learning Machine Learning and AI Testing from ChatGPT",
           candidate_labels=["education", "marketing", "motivation", "business"])

## NER Classification
"""
- NER = Named Entity Recognition → the task of finding and classifying entities in text (like names, organizations, locations, emails, etc.).
- NER extracts and labels important real-world entities from text.
- NER classifies tokens/entities, not whole text.
- Common Entity Types
    PERSON → John, Alice
    ORG → Google, OpenAI
    LOC / GPE → India, New York
    DATE → 2025, Monday
    MONEY → $100, ₹500
- Hugging Face loads a default pre-trained NER model (usually dslim/bert-base-NER).
- grouped_entities=True → merges tokens that belong to the same entity into one result.
Without this, you'd get fragmented outputs (e.g., “Kar” + “thik” instead of “Karthik”).
"""
classifier = pipeline("ner", grouped_entities=True)
classifier("Karthik is working in ExecuteAutomation and living in NZ and my email is karthik@ea.com")


### Testing LLM using HF Evaluate
"""
Precision → Of all predicted positives, how many are actually correct.
Recall → Of all actual positives, how many were correctly identified.
F1-score → Harmonic mean of precision and recall (balance between both).
Accuracy → Overall percentage of correct predictions (both positive and negative).
"""
## Accuracy
import evaluate # HF library for evaluation metrics
accuracy = evaluate.load("accuracy") # loads the built‑in accuracy metric.
references = [1,0,1,1] # ground truth labels, the correct answers.
predications = [1,0,0,0]
results = accuracy.compute(references=references, predictions=predications)
print(results) # {'accuracy': 0.5}
print(f"Accuracy: {results['accuracy']:.2%}")  # Accuracy: 50.00%  

## Precision (Exact Match)
exact_match = evaluate.load("exact_match")
references = ["Execute", "Automation"]
predications = ["Execute", "Automation"]
results = exact_match.compute(references=references, predictions=predications)
print(f"Exact Match: {results['exact_match']:.2%}") # Exact Match: 100.00%

## F1-Score
f1 = evaluate.load("f1")
references = [1,0,1,1,0,1,0,1,1]
predications = [1,0,1,1,1,1,1,0,1]
results = f1.compute(references=references, predictions=predications, average="binary")
print(f"F1-Score: {results['f1']:.2%}")
"""
- In Hugging Face's evaluate library, the average parameter controls how the F1 score is calculated across classes.
    - When you set average="binary", the F1 score is computed only for the positive class (label = 1).
    - This is the default for binary classification tasks, where you care about how well the model identifies positives.
    - It ignores the negative class (label = 0) when calculating precision, recall, and F1.
- Other options for average
    - "micro" → Calculates metrics globally by counting total true positives, false negatives, and false positives.
    - "macro" → Calculates metrics for each class independently, then takes the unweighted average.
    - "weighted" → Same as macro, but weights each class's score by its support (number of true instances).
    - None → Returns F1 for each class separately instead of averaging.
"""

## Using HF Evaluate Method to Evalute a LLM using Pipeline function 🤖⚙️
sentiment_pipeline = pipeline("sentiment-analysis")
accuracy = evaluate.load("accuracy")
datasets = [
    {"text": "I love learning ML and AI in Testing", "label": 0},
    {"text": "I hate working with a machine which has got no GPU for my AI training", "label": 0},
    {"text": "I like driving fast cars", "label": 1}
]
# "label" → the expected sentiment (0 or 1).
predications = sentiment_pipeline([data["text"] for data in datasets])
print(predications)
"""
[{'label': 'POSITIVE', 'score': 0.9988466501235962},
 {'label': 'NEGATIVE', 'score': 0.9995275735855103},
 {'label': 'POSITIVE', 'score': 0.9991025924682617}]
"""
prediction_labels = [1 if pred["label"] == "POSITIVE" else 0 for pred in predications]
true_labels = [dataset['label'] for dataset in datasets]
print(prediction_labels, true_labels)  # [1, 0, 1] [0, 0, 1]
results = accuracy.compute(predictions=prediction_labels, references=true_labels) # compute accuracy
print(f"Accuracy: {results['accuracy']:.2%}") # Accuracy: 66.67%
# same can be done for Precision, Recall, F1-Score using HF evaluate library
"""==================================================================================================="""

# Section 10 and 11, is not completed in details. Below are the some important points mentioned.
from deepeval.tracing import (
    observe,
    update_current_span
)
"""
# Why use `deepeval.tracing`?
* DeepEval tracing provides tools to **observe and log** the execution of LLMs, chains, agents, and tools.
* It helps QA teams **track and debug** complex AI workflows by capturing detailed traces of operations
* **`observe`** → Decorator to **trace and log execution** of LLM calls, chains, agents, or tools for evaluation and debugging.
* **`update_current_span`** → Adds **custom metadata** (inputs, outputs, tags, errors) to the current trace/span.
# QA Use Cases
* Track **prompt → retrieval → tool → LLM** flow
* Debug **hallucinations and failures**
* Send rich traces to **Confident AI** for review

“Tracing affects model behavior”
 ✅ NO: It only **observes and logs**, not change outputs.

"""

@observe(type='llm', model='qwen2.5:latest')
def local_llms(): # To use the @observe decorator, the function must take no arguments and function is needed.
    return ChatOllama(
        base_url="http://localhost:11434",
        model = "qwen2.5:latest",
        temperature=0.5,
        max_tokens = 250
    )
    # return ChatOpenAI(model="gpt-4.1-2025-04-14", max_completion_tokens=300)
    
llm = local_llms()

@tool
@observe(type='tool')
def add_numbers(a: int, b: int) -> int:
    "Add two numbers and return results."
    result = int(a) + int(b)  # Earlier we were reuturning int, but that Confident AI needs to be a String or json. 
    return f"The sum of {a} and {b} is {result}"

# We have some predefoinied observe types like 'llm', 'tool', 'agent' etc. But if not passed, then it will be 'custom'.
@observe(type='agent', available_tools=["add_numbers", "subtract_numbers", "search_tool"], metrics=[ToolCorrectnessMetric()])
def main_ai_agent(query):
    agent = initialize_agent(
        tools= tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        return_intermediate_steps=True
    )
    
    response = agent.invoke(query)
    
    # The use of below update_current_span is optional, but it helps in associating the test case with the current execution span for better traceability.
    update_current_span(
        test_case=LLMTestCase(
            input=query,
            tools_called=[ToolCall(name="add_numbers")],
            expected_tools=[ToolCall(name="add_numbers")],
            actual_output=response['output']
        )
    )
    
    return response




















