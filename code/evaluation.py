"""
Measure answer quality with RAGAS. RAGAS uses Gemini as a "judge" and gives scores from 0 to 1:

- faithfulness       Is the answer backed by the retrieved chunks? (low = hallucination)
- answer relevancy   Does the answer actually address the question?
- context precision  Are the retrieved chunks relevant and ranked well?
- context recall     Did we retrieve everything needed for the correct answer?

Before running: edit TEST_QUESTIONS so they match YOUR PDFs.
Run from the mediguide-ai folder:
    python code/evaluation.py
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # RAGAS prints many of these

from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall

from config import llm, embeddings, DATA_DIR
from supervisor import ask

# Questions + the correct answers ("reference"), written by you from your documents
TEST_QUESTIONS = [
    {
        "question": "What are the common symptoms of type 2 diabetes?",
        "reference": "Common symptoms include increased thirst, frequent urination, tiredness, blurred vision and slow-healing wounds.",
    },
    {
        "question": "What is the first-line medication for type 2 diabetes?",
        "reference": "Metformin is the first-line medication for type 2 diabetes, along with lifestyle changes.",
    },
    {
        "question": "What blood pressure value is considered hypertension?",
        "reference": "Hypertension is diagnosed when blood pressure is 140/90 mmHg or higher on repeated measurements.",
    },
]

# 1. Run the agent on every test question and collect what RAGAS needs
samples = []
for item in TEST_QUESTIONS:
    print("Asking:", item["question"])
    result = ask(item["question"])
    samples.append({
        "user_input": item["question"],
        "response": result["answer"],
        "retrieved_contexts": [s["text"] for s in result["sources"]] or [""],
        "reference": item["reference"],
    })

# 2. Score them. max_workers=2 keeps us under Gemini's free-tier rate limit.
results = evaluate(
    dataset=EvaluationDataset.from_list(samples),
    metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
    llm=LangchainLLMWrapper(llm),
    embeddings=LangchainEmbeddingsWrapper(embeddings),
    run_config=RunConfig(max_workers=2, timeout=180),
)

# 3. Show average scores and save per-question scores
print("\n===== RAGAS SCORES =====")
print(results)
results.to_pandas().to_csv(DATA_DIR / "ragas_results.csv", index=False)
print(f"Per-question scores saved to {DATA_DIR / 'ragas_results.csv'}")
