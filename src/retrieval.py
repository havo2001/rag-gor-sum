import numpy as np
import torch
import re
import faiss
from transformers import AutoTokenizer, AutoModel
from helper import show_time
from langchain_text_splitters import TokenTextSplitter
from rank_bm25 import BM25Okapi


# -----------------------------------------------------
# Dense retriever, use for GoR also
# -----------------------------------------------------
def get_dense_retriever(retriever):
    # Load the dense retriever
    if retriever == 'contriever':
        query_tokenizer = ctx_tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        query_encoder = ctx_encoder = AutoModel.from_pretrained('facebook/contriever')
    else:
        raise Exception("Retriever Error")
    
    return query_tokenizer, ctx_tokenizer, query_encoder, ctx_encoder


def split_batch(instructions, batch_size):
    batch_instructions = []
    sub_batch = []
    for idx, ins in enumerate(instructions):
        if idx != 0 and idx % batch_size == 0:
            batch_instructions.append(sub_batch)
            sub_batch = [ins]
        else:
            sub_batch.append(ins)

    if len(sub_batch) != 0:
        batch_instructions.append(sub_batch)
    
    return batch_instructions


def get_dense_embedding(instructions, retriever, tokenizer, model, trunc_len=512, batch_size=64):
    emb_list = []
    batch_instructions = split_batch(instructions=instructions, batch_size=batch_size)
    for sub_batch in batch_instructions:
        if retriever == 'contriever':
            inputs = tokenizer(sub_batch, 
                               padding=True, 
                               truncation=True, 
                               return_tensors='pt', 
                               max_length=trunc_len).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Mean pooling
            token_embeddings = outputs[0].masked_fill(~inputs['attention_mask'][..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / inputs['attention_mask'].sum(dim=1)[..., None]

            for emb in sentence_embeddings:
                emb_list.append(emb)
        else:
            raise Exception('Error')
    
    return emb_list


def dense_neighborhood_search(corpus_data, query_data, metric='ip', num=8):
    # This function is like search and can work with multiple queries
    # I think we can combine this function with the function below
    xq = torch.vstack(query_data).cpu().numpy()
    xb = torch.vstack(corpus_data).cpu().numpy()
    dim = xb.shape[1]

    if metric == 'l2':
        index = faiss.IndexFlatL2(dim)
    elif metric == 'ip':
        index = faiss.IndexFlatIP(dim)
        xq = xq.astype('float32')
        xb = xb.astype('float32')
        faiss.normalize_L2(xq)
        faiss.normalize_L2(xb)
    else:
        raise Exception("Index Metric Not Exist")
    index.add(xb)
    D, I = index.search(xq, num)

    return I[0]


def run_dense_retrieval(query_embedding, ch_text_chunk_embed, ch_text_chunk, chunk_num=4):
    print("{} Dense Retrieval...".format(show_time()))
    retrieved_idx = dense_neighborhood_search(ch_text_chunk_embed, query_embedding, num=chunk_num)
    retrieved_idx = list(retrieved_idx)

    print("{} Retrieved Chunks:".format(show_time()), retrieved_idx)
    retrieved_chunks = []
    for idx in retrieved_idx:
        retrieved_chunks.append(ch_text_chunk[idx])

    return retrieved_idx, retrieved_chunks


# -----------------------------------------------------
# Sparse retriever, use for baseline only
# -----------------------------------------------------
def run_sparse_retrieval(retriever, query: str, text_chunk: list[str], k1=1.2, b=0.75, chunk_num=6):
    def _simple_tokenize(text: str):
        return re.findall(r"\w+", text.lower())
    if retriever == 'bm25':  
        tok = _simple_tokenize
        tok_chunk = [tok(t) for t in text_chunk]
        bm25 = BM25Okapi(tok_chunk, k1=k1, b=b)
        tok_query = tok(query)
        scores = bm25.get_scores(tok_query)
        order = np.argsort(scores)[::-1][:chunk_num]
        retrieved_idx = order.tolist()
        retrieved_chunks = [text_chunk[i] for i in retrieved_idx]
    else:
        raise Exception("Retriever Error")
    
    return retrieved_idx, retrieved_chunks