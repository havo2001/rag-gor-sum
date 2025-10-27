import re
import numpy as np
from rank_bm25 import BM25Okapi


class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b

    @staticmethod
    def _tokenize(text: str):
        return re.findall(r"\w+", text.lower())
    
    def retrieve(self, query, text_chunk, chunk_num=6):
        tok_chunk = [self._tokenize(t) for t in text_chunk]
        tok_query = self._tokenize(query)
        
        bm25 = BM25Okapi(tok_chunk, k1=self.k1, b=self.b)
        scores = bm25.get_scores(tok_query)
        order = np.argsort(scores)[::-1][:chunk_num]
        retrieved_idx = order.tolist()
        retrieved_chunks = [text_chunk[i] for i in retrieved_idx]

        return retrieved_idx, retrieved_chunks