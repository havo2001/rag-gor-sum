# Keep the prompt the same as the original paper
QUERY_PROMPT_QMSUM_NORMAL = """
Refer to the following meeting transcripts and answer the question. 

SUPPORTING MATERIALS:
{materials}

QUESTION:
{question}
"""

QUERY_PROMPT_NORMAL = {
    'qmsum': QUERY_PROMPT_QMSUM_NORMAL
}

QUERY_PROMPT_QMSUM = """
Refer to the following meeting transcripts and answer the question with brief but complete explanations. 

SUPPORTING MATERIALS:
{materials}

QUESTION:
{question}
"""

QUERY_PROMPT = {
    "qmsum": QUERY_PROMPT_QMSUM
}


QUERY_GENERATE = """
You are a great questioner of any text, and are adept at asking valuable and insightful questions. 
Your goal is to generate 1 summary question for the text provided below. 
The generated summary question should try to simulate the tone of human questions as much as possible, 
and make sure that the generated question must be interrogative sentences and a summary question. 
Important! Please make sure this text must be a complete and non-redundant answer to the generated summary question. 
Please directly output the generated summary question, do not output irrelevant text.

DOCUMENT:
{document}
"""


LLM_AS_JUDGE_PROMPT = """
You are an expert evaluator of meeting summaries. Your job is to compare two candidate summaries (Answer A and Answer B) for the same question.

Rules:
- Be strictly unbiased: do NOT favor the first/second answer, and do NOT favor longer answers.
- If a REFERENCE SUMMARY is provided, use it to judge coverage and correctness.
- If something is unclear, prefer "tie" rather than guessing.
- Return ONLY valid JSON. No markdown. No extra text.

QUESTION:
{QUERY}

REFERENCE SUMMARY (optional):
{GROUND_TRUTH}

ANSWER A:
{ANSWER_A}

ANSWER B:
{ANSWER_B}

Evaluate and output JSON EXACTLY in this schema:
{
  "faithfulness_winner": "A" | "B" | "tie",
  "coverage_winner": "A" | "B" | "tie",
  "coherence_winner": "A" | "B" | "tie",
  "conciseness_winner": "A" | "B" | "tie",
  "overall_winner": "A" | "B" | "tie",
  "rationale": [
    "short bullet 1",
    "short bullet 2",
    "short bullet 3"
  ]
}

"""