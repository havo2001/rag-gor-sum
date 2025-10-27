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