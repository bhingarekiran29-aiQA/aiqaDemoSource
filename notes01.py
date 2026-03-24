### DeepEval Introduction
# %pip install -U deepeval
# %pip install dotenv
# %pip install litellm

"""
# DeepEval is an open-source evaluation framework used to test, measure, and validate LLM applications such as RAG systems, agents, 
and prompt pipelines.
# Traditional API automation validates fixed, predictable responses, while DeepEval evaluates the quality and correctness of variable, 
probabilistic LLM outputs.
# DeepEval helps automate LLM quality evaluation, but it complements—not replaces—human judgment and traditional testing.
# I first use API automation to validate stability and contracts, then apply DeepEval on the response content to validate semantic quality 
and hallucinations.
# DeepEval brings automation and metrics to LLM quality testing, but human judgment is still essential.
---
# Why DeepEval is used
    * Automates LLM quality testing
    * Converts subjective LLM behavior into measurable metrics
    * Fits well into CI/CD pipelines
---
# What DeepEval evaluates
    * Answer correctness
    * Relevance
    * Faithfulness (hallucination detection)
    * Context usage (RAG)
    * Tool / agent behavior
---
# Common DeepEval Metrics
    * Answer Relevancy
    * Faithfulness
    * Context Precision
    * Context Recall
    * Hallucination
    * Bias / Toxicity (optional)
---
# QA Use Cases
    * Prompt regression testing
    * RAG accuracy validation
    * Hallucination detection
    * Agent output validation
    * Model version comparison
---
# Where it fits in the LLM Stack
LLM / RAG / Agent
        ↓
     DeepEval
        ↓
Quality Metrics & Pass/Fail
---
# Real Project Flow (Interview Gold)
User Query
   ↓
API Automation (stability)
   ↓
DeepEval (quality metrics)
   ↓
Manual Review (critical cases)
---
## Interview Trap (⚠️)
❌ “DeepEval is a monitoring tool”
✅ It's primarily an evaluation & testing framework
---
# DeepEval Interview Questions
- Hallucination? : When LLM generates unsupported or fabricated info. Means answer not grounded in context. Means risk of wrong info.
- What is faithfulness? : How well the answer sticks to provided context.
- How does DeepEval detect hallucination? : By comparing answers against source context using evaluation metrics.
- Can DeepEval guarantee correctness?: No, it reduces risk but doesn't eliminate errors.
- DeepEval limitations? : Scores are probabilistic and model-dependent.
- Integration with CI/CD? : Via API calls in test pipelines.
- Custom metrics? : Yes, users can define their own evaluation metrics.
- Supported models? : OpenAI, HuggingFace, Ollama, etc.
- Replace manual QA? : No, it complements manual testing.
"""
import deepeval
deepeval.login("confident_us_8k9P7QpyyKgjpa7yzXG0ULlki3JAwq0DPAstgNKA1x0=")

from dotenv import load_dotenv
load = load_dotenv('./../.env')

# Writing simple DeepEval Test
import os
os.environ["OPENAI_API_KEY"] = ""  # Clear OpenAI key
os.environ["DEEPEVAL_MODEL"] = "ollama/llama3.2:latest" # Set default model to Ollama

# Answer Relevancy Metrics - Standalone
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from deepeval.models import GPTModel
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import ContextualPrecisionMetric

# model = GPTModel(model="gpt-3.5-turbo") # Free Quota exhausted for this model. So using ollama. 
# answer_relevancy_metric = AnswerRelevancyMetric() # It will select model automatically based on Open AI API key or Ollama. 
ollama_model = OllamaModel(model="llama3.2:latest")
answer_relevancy_metric = AnswerRelevancyMetric(model=ollama_model)
contextual_precision_metrics = ContextualPrecisionMetric(model = ollama_model)

test_case = LLMTestCase(
  input="Who is the current president of the United States of America?",
  actual_output="Joe Biden",
  retrieval_context=["Joe Biden serves as the current president of America."],
  expected_output="Donald Trump is the current president of America." # Added for contextual_precision_metrics
)

answer_relevancy_metric.measure(test_case)
contextual_precision_metrics.measure(test_case=test_case)
print(answer_relevancy_metric.score, contextual_precision_metrics.score)
print(contextual_precision_metrics.success, contextual_precision_metrics.score_breakdown)

"""
- input → the question asked to the LLM
- actual_output → the LLM's answer
- retrieval_context → the information your system retrieved to help answer the question.
- expected_output → the correct answer you expect from the LLM
----------------------------------------------------------------------------------------
- answer_relevancy_metric → Measures how well the answer addresses the user's question.
- contextual_precision_metrics → Checks whether only relevant context was used in the answer.
- faithfulness_metric → Checks if the answer is grounded in the provided context.
- context_recall_metric → Measures how much of the relevant context was used in the answer.
- hallucination_metric → Detects unsupported or fabricated information in the response.
- answer_correctness_metric → Compares the answer against an expected/reference output.
- bias_metric → Detects biased or unfair language in the output.   
- toxicity_metric → Identifies harmful or unsafe content.
- consistency_metric → Measures output stability across multiple runs.
- tool_correctness_metric → Validates whether the agent used the correct tool.
----------------------------------------------------------------------------------------
- score → a numerical value (0 to 1) indicating performance on the metric.
- success → a boolean indicating if the test passed based on a threshold. It is default 0.5, but can be changed while creating metric object.
- score_breakdown → detailed insights into how the score was calculated.
----------------------------------------------------------------------------------------
- input and actual_output are manadatory fields.
- Other fileds are optional and choosed based on the metric used.
| Metric            | Required Fields                    |
| ----------------- | ---------------------------------- |
| Relevance         | input, actual_output               |
| Faithfulness      | actual_output, retrieval_context   |
| Context Recall    | retrieval_context                  |
| Context Precision | expected_output, retrieval_context |
| Hallucination     | actual_output, retrieval_context   |
--------------------------------------------------------------------------  ----------- 
# Key Metric Groupings for Interviews
Prompt → Relevance + Correctness
RAG → Faithfulness + Context
Agent → Tool + Decision Logic
-------------------------------------------------------------------------------------------------------
# Real interview scenario answers:
1. A prompt change caused user complaints but API tests are passing. What do you do?
> I run DeepEval relevance and consistency metrics on old vs new prompts, compare score drops, and roll back if regression exceeds thresholds.
2. Users report hallucinated answers in a RAG system. What's your approach?
> I validate faithfulness and context recall, test with missing or noisy documents, and tune retrieval or chunking based on metric failures.
3. DeepEval scores are high but answer is factually wrong.
> High scores indicate grounding, not factual truth, so I add answer correctness checks and manual review for critical flows.
4. Agent gives correct answer but uses wrong tool.
> I validate tool correctness and agent decision logic, not just the final output.
5. Should DeepEval block production deployments?
> Only for major regressions; otherwise it should act as a quality signal, not a hard blocker.
6. LLM responses vary on every run. How do you test?
> I evaluate score trends across multiple runs instead of exact output matching.
7. How do you test before upgrading an LLM model?
> I run the same DeepEval test suite on both models and compare average metric scores. 
8. What if retrieval context is incomplete?
> Faithfulness should drop; if it doesn't, it indicates hallucination risk.
9. How do you explain DeepEval results to non-technical teams?
> I present metric trends and quality improvements instead of raw scores.
10. Can DeepEval replace human testing?
> No, it automates semantic validation but critical cases still need human review.
"""
test_case1, test_case2 = "", ""
# Evaluate our Tests without Standalone using - Evaluate
from deepeval.evaluate import evaluate
evaluate(test_cases=[test_case1], metrics=[answer_relevancy_metric])
evaluate(test_cases=[test_case1, test_case2], metrics=[answer_relevancy_metric])
"""
# What evaluate() Does
- Executes selected DeepEval metrics
- Scores each test case
- Determines pass/fail based on thresholds
- Uploads results to Confident AI Portal
-----------------------------------------------
# Confident AI Portal (Important for Interview)
- Centralized dashboard for: Test runs, Metric scores, Regression tracking
- Auto-opens test run link after execution
--------------------------------------------------------------------------
- this evaluate method will send the result to Confident AI Portal
- It will automatically open the link like this https://app.confident-ai.com/project/cmisxih9601ubnp1fpqabp0d.......
--------------------------------------------------------------------------------------------------------------------
- Golden → one trusted reference test case
- Golden Dataset → a logical collection of Goldens
- EvaluationDataset → The DeepEval class that stores the Golden dataset
------------------------------------------------------------------------
"""
# Evaluate With Golden DataSet and EvaluationDataSet
from deepeval.dataset import EvaluationDataset, Golden

# Create Golden instead of Test cases
golden = Golden(
    input="Who is the current president of the United States of America?",
    expected_output="Joe Biden",
    context=["Joe Biden serves as the current president of America."] 
)
dataset = EvaluationDataset()
dataset.add_golden(golden)

# Creating Test Case from Golden
from deepeval.models import OllamaModel
ollama_model = OllamaModel(model="llama3.2:latest")

for golden in dataset.goldens:
    test_case = LLMTestCase(
        input=golden.input,
        expected_output=golden.expected_output,
        actual_output="Joe Biden", # Simulated LLM output
        retrieval_context=golden.context
    )
    
    dataset.add_test_case(test_case) # Adding test case to dataset
evaluate(test_cases=dataset.test_cases, metrics=[AnswerRelevancyMetric(model = ollama_model)]) # Evaluating all test cases in dataset

### Creating Evaluation Dataset as Goldens in Confident AI
test_data = [
    {
        "input": "Who is the current president of the United States of America?",
        "expected_output": "Joe Biden",
    },
    {
        "input": "Who introducted the GPT Model?",
        "expected_output": "Open AI"
    }
]

new_goldens = []

for data in test_data:
    golden = Golden(
        input= data['input'],
        expected_output=data['expected_output'],
    )
    new_goldens.append(golden)
    
new_dataset = EvaluationDataset(goldens=new_goldens)
new_dataset.push(alias="TestGoldenDataSet", finalized=False) # Pushing new golden dataset to Confident AI Portal
# Response: ✅ Dataset successfully pushed to Confident AI! View at 
# https://app.confident-ai.com/project/cmisxih9601ubnp1fpqabp0df/datasets/cmiv9frhl01p6n11ftkn0x8x1

### Pull the Dataset from Confident AI
cloudDataSet = EvaluationDataset()
cloudDataSet.pull(alias="TestGoldenDataSet")
evaluate(test_cases=cloudDataSet.test_cases, metrics=[AnswerRelevancyMetric(model = ollama_model)]) 
# Evaluating all test cases in dataset pulled from Confident AI Portal
"""
- Also Confident AI UI allows direct creation and management of Golden datasets.
- “Confident AI is only a reporting dashboard” --> ✅ Wrong — it's also a test data management system
# Real-World QA Flow (Very Interview-Friendly)
        QA creates Goldens (UI or Code)
                ↓
        Stored in Confident AI
                ↓
        Pulled into CI/CD or local tests
                ↓
        Evaluated on new models/prompts
-------------------------------------------------------------------------------
# LangSmith: Option to DeepeEval. A LangChain-centric observability platform for debugging, tracing, and evaluating LLM workflows.
📌 Interview Trap Alerts
❌ “Confident AI is just DeepEval UI.”
✅ NO: It's a full evaluation platform with dashboard, dataset editor, regression tracking, and observability. 
❌ “LangSmith is only for logging.”
✅ NO: It also supports dataset evaluation, prompt versioning, and scenario testing.
"""