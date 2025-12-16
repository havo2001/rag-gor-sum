from transformers import AutoTokenizer, AutoModel
from src.helper import show_time
import torch
import faiss

from rank_bm25 import BM25Okapi


class Contriever:
    def __init__(self, device, metric='ip', batch_size=64):
        self.query_tokenizer = self.ctx_tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        self.query_encoder = self.ctx_encoder = AutoModel.from_pretrained('facebook/contriever')
        self.query_encoder.eval()
        self.ctx_encoder.eval()
        self.metric = metric
        self.batch_size = batch_size


    def split_batch(self, instructions):
        batch_instructions = []
        sub_batch = []
        for idx, ins in enumerate(instructions):
            if idx != 0 and idx % self.batch_size == 0:
                batch_instructions.append(sub_batch)
                sub_batch = [ins]
            else:
                sub_batch.append(ins)

        if len(sub_batch) != 0:
            batch_instructions.append(sub_batch)
        
        return batch_instructions
    
    @torch.inference_mode()
    def get_dense_embedding(self, instructions, tokenizer, model, trunc_len=512):
        emb_list = []
        batch_instructions = self.split_batch(instructions=instructions)
        for sub_batch in batch_instructions:
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
        
        return emb_list
    

    def dense_neighborhood_search(self, corpus_data, query_data, chunk_num=6):
        # This function is like search and can work with multiple queries
        # I think we can combine this function with the function below
        xq = torch.vstack(query_data).cpu().numpy()
        xb = torch.vstack(corpus_data).cpu().numpy()
        dim = xb.shape[1]

        if self.metric == 'l2':
            index = faiss.IndexFlatL2(dim)
        elif self.metric == 'ip':
            index = faiss.IndexFlatIP(dim)
            xq = xq.astype('float32')
            xb = xb.astype('float32')
            faiss.normalize_L2(xq)
            faiss.normalize_L2(xb)
        else:
            raise Exception("Index Metric Not Exist")
        index.add(xb)
        D, I = index.search(xq, chunk_num)

        return I[0]
    
    def retrieve(self, query, text_chunk, chunk_num):
        # Make sure the query are list of strings
        if not isinstance(query, list):
            query = [query]


        print("{} Dense Retrieval...".format(show_time()))
        text_chunk_emb = self.get_dense_embedding(instructions=text_chunk, tokenizer=self.ctx_tokenizer, model=self.ctx_encoder)
        query_emb = self.get_dense_embedding(instructions=query, tokenizer=self.query_tokenizer, model=self.query_encoder)
        retrieved_idx = self.dense_neighborhood_search(text_chunk_emb, query_emb, chunk_num=chunk_num)
        retrieved_idx = list(retrieved_idx)

        print("{} Retrieved Chunks:".format(show_time()), retrieved_idx)
        retrieved_chunks = []
        for idx in retrieved_idx:
            retrieved_chunks.append(text_chunk[idx])

        return retrieved_idx, retrieved_chunks
