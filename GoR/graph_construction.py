import torch
import os
import argparse
import networkx as nx
import dgl
import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import TokenTextSplitter

from src.contriever import Contriever

from src.helper import show_time, set_seed, check_path, store_nx, write_to_pkl
from src.llm import get_llm_response_via_api
from src.data_process import split_corpus_into_chunk, get_processed_data

from prompt.prompt import QUERY_GENERATE, QUERY_PROMPT



def rag_retrieval(chunk_list, rag_query, retriever, chunk_num):
    if len(chunk_list) <= chunk_num:
        return chunk_list

    _, retrieved_text_list = retriever.retrieve(query=rag_query, text_chunk=chunk_list, chunk_num=chunk_num)
    return retrieved_text_list


def mem_retrieval(
    mem_chunk_embedding,
    all_doc_chunk_list,
    all_doc_chunk_list_embedding,
    rag_query,
    graph,
    retriever,
    recall_chunk_num,
):
    
    # Collect memory node texts from graph
    mem_chunk_list = [node for node, _ in graph.nodes(data=True)]


    assert len(mem_chunk_embedding) == len(mem_chunk_list), (
        f"{len(mem_chunk_embedding)} != {len(mem_chunk_list)}"
    )

    # Ensure we have embeddings for the full document chunks
    if all_doc_chunk_list_embedding is None:
        all_doc_chunk_list_embedding = retriever.get_dense_embedding(
            instructions=all_doc_chunk_list,
            tokenizer=retriever.ctx_tokenizer,
            model=retriever.ctx_encoder
        )

    assert len(all_doc_chunk_list_embedding) == len(all_doc_chunk_list)

    # Merge: memory nodes + any doc chunks not already in memory
    mem_set = set(mem_chunk_list)  # faster membership checks
    mem_chunk_embedding_copy = list(mem_chunk_embedding)  # shallow copy

    for chunk_text, chunk_emb in zip(all_doc_chunk_list, all_doc_chunk_list_embedding):
        if chunk_text not in mem_set:
            mem_chunk_list.append(chunk_text)
            mem_chunk_embedding_copy.append(chunk_emb)
            mem_set.add(chunk_text)

    # Embed the query (must be a list of 1 string)
    rag_query_embedding = retriever.get_dense_embedding(
        instructions=[rag_query],
        tokenizer=retriever.query_tokenizer,
        model=retriever.query_encoder
    )
    assert len(rag_query_embedding) == 1

    # Device alignment (avoid CPU/GPU/MPS mismatch errors)
    q_dev = rag_query_embedding[0].device
    mem_chunk_embedding_copy = [e.to(q_dev) for e in mem_chunk_embedding_copy]

    # Retrieve top-K indices over the merged corpus
    retrieved_index = retriever.dense_neighborhood_search(
        corpus_data=mem_chunk_embedding_copy,
        query_data=rag_query_embedding,
        chunk_num=recall_chunk_num
    )
    retrieved_index = list(retrieved_index)

    retrieved_text_list = [mem_chunk_list[i] for i in retrieved_index]
    return retrieved_text_list, retrieved_index


def get_node_embedding_list(dgl_graph):
    mem_chunk_embedding = dgl_graph.ndata['feat']
    mem_chunk_embedding = [i for i in mem_chunk_embedding]

    return mem_chunk_embedding


def record_graph_construction(
    query,
    support_materials,     # List[str] retrieved chunk texts
    response,              # str response text
    graph,                 # nx.Graph (nodes are strings)
    dgl_graph,             # dgl.graph
    training_data,         # List[dict]
    retriever,             # your Contriever() instance
    answer=None
):
    """
    Adds nodes (response + retrieved chunks) into NX graph and DGL graph.
    Adds edges chunk->response.
    Stores training_data entries with DGL node indices.
    """

    sub_training_data = {"query": query}
    if answer is not None:
        sub_training_data["answer"] = answer

    # Existing node list in insertion order as seen by NX iteration
    existing_chunks = [node for node, _ in graph.nodes(data=True)]

    # Track new nodes to embed + add to DGL
    non_dup_chunks = []

    # Add response node if new
    if response not in graph:
        graph.add_node(response)
        non_dup_chunks.append(response)
        existing_chunks.append(response)

    # Add retrieved chunk nodes if new
    for chunk in support_materials:
        if chunk not in graph:
            graph.add_node(chunk)
            non_dup_chunks.append(chunk)
            existing_chunks.append(chunk)

    # Build chunk_id_map aligned with existing_chunks list
    # IMPORTANT: this must match DGL node order
    chunk_id_map = {chunk: idx for idx, chunk in enumerate(existing_chunks)}

    # If there are new nodes, embed them and append to DGL graph
    if len(non_dup_chunks) > 0:
        new_node_embedding = retriever.get_dense_embedding(
            instructions=non_dup_chunks,
            tokenizer=retriever.ctx_tokenizer,
            model=retriever.ctx_encoder
        )
        dgl_graph.add_nodes(
            num=len(non_dup_chunks),
            data={"feat": torch.vstack(new_node_embedding).cpu()}
        )

    # Response node index in DGL
    sub_training_data["response"] = [chunk_id_map[response]]

    # Raw support node indices in DGL
    sub_training_data["raw"] = []

    for chunk in support_materials:
        sub_training_data["raw"].append(chunk_id_map[chunk])

        # Add NX edge chunk -> response
        if not graph.has_edge(chunk, response):
            graph.add_edge(chunk, response, weight=1)

        # Add DGL edge chunk -> response
        u = chunk_id_map[chunk]
        v = chunk_id_map[response]
        if not dgl_graph.has_edges_between(u, v):
            dgl_graph.add_edges(u, v, data={"w": torch.ones(1, 1)})

    training_data.append(sub_training_data)
    return graph, dgl_graph, training_data


def llm2query(prompt, api_key, llm_model, tau=0.5, seed=42):
    content, usage = get_llm_response_via_api(
                                                prompt=prompt,
                                                API_KEY=api_key,
                                                LLM_MODEL=llm_model,
                                                TAU=tau,
                                                SEED=seed,
                                                )
    content = content.split("\n")
    for ind, c in enumerate(content):
        for start_ind in range(len(c)):
            if str(c[start_ind]).isalpha():
                break
        content[ind] = c[start_ind:]

    return [i for i in content if len(i.strip()) != 0], usage


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument('--train', action='store_true')
    parser.add_argument("--llm_model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default='mps')
    parser.add_argument("--tau", type=float, default=0)
    parser.add_argument("--query_tau", type=float, default=0.5)
    parser.add_argument("--retriever", type=str, default="contriever")
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--chunk_overlap", type=int, default=32)
    parser.add_argument("--recall_chunk_num", type=int, default=6)
    parser.add_argument("--query_num", type=int, default=30)
    opt = parser.parse_args()
    DATASET = opt.dataset
    TRAIN = opt.train
    LLM_MODEL = opt.llm_model
    SEED = opt.seed
    DEVICE = opt.device
    TAU = opt.tau
    QUERY_TAU = opt.query_tau
    RETRIEVER = opt.retriever
    CHUNK_SIZE = opt.chunk_size
    CHUNK_OVERLAP = opt.chunk_overlap
    RECALL_CHUNK_NUM = opt.recall_chunk_num
    QUERY_NUM = opt.query_num

    set_seed(int(SEED))
    load_dotenv()

    retriever = Contriever(device=DEVICE)

    TEXT_SPLITTER = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    data = get_processed_data(dataset=DATASET, train=TRAIN)
    print("{} #Data: {}".format(show_time(), len(data)))
    MAX_NUM = 400 if TRAIN else 30
    data = data[:MAX_NUM]
    check_path("./graph")
    
    # Initialize token usage tracking
    total_query_gen_input_tokens = 0
    total_query_gen_output_tokens = 0
    total_response_input_tokens = 0
    total_response_output_tokens = 0
    
    for ind, sample in enumerate(data):
        if ind <= 7:
            continue # crashed after creating the 8th graph so continue from the 8th
        # Due to budget constraints, we randomly select at most 400 samples for training and 30 samples for evaluation.
        # You can optionally create a dev set for hyper-parameter tuning
        all_doc_chunk_list = split_corpus_into_chunk(dataset=DATASET, sample=sample, text_splitter=TEXT_SPLITTER)
        all_doc_chunk_list_embedding = retriever.get_dense_embedding(instructions=all_doc_chunk_list, tokenizer=retriever.ctx_tokenizer, model=retriever.ctx_encoder)
        graph = nx.Graph()
        dgl_graph = dgl.graph(([], []), num_nodes=0)
        training_data = []
        # Query Simulation
        user_question = []
        user_answer = []
        while len(user_question) < QUERY_NUM:
            unsup_answer = np.random.choice(all_doc_chunk_list, size=1, replace=False)[0].split()
            unsup_answer = " ".join(unsup_answer)
            gen_q_list, usage = llm2query(prompt=QUERY_GENERATE.format_map({"document": unsup_answer}), api_key=os.getenv("API_KEY"), llm_model=LLM_MODEL, tau=QUERY_TAU, seed=SEED)
            
            # Track query generation token usage
            if usage:
                total_query_gen_input_tokens += getattr(usage, 'prompt_tokens', 0)
                total_query_gen_output_tokens += getattr(usage, 'completion_tokens', 0)
            
            if len(gen_q_list) > 0:
                gen_q = gen_q_list[0]
                if gen_q not in user_question:
                    user_question.append(gen_q)
                    user_answer.append(unsup_answer)
                    print("{} Generate Query {}/{}:\n{}".format(show_time(), len(user_question), QUERY_NUM, gen_q))
        # Graph Construction
        for uid, user_query in enumerate(user_question):
            if graph.number_of_nodes() == 0:
                retrieved_chunks = retrieved_chunks = rag_retrieval(chunk_list=all_doc_chunk_list, rag_query=user_query, retriever=retriever, chunk_num=RECALL_CHUNK_NUM)

            else:
                mem_chunk_embedding = get_node_embedding_list(dgl_graph=dgl_graph)
                retrieved_chunks, _ = mem_retrieval(
                                                    mem_chunk_embedding=mem_chunk_embedding,
                                                    all_doc_chunk_list=all_doc_chunk_list,
                                                    all_doc_chunk_list_embedding=all_doc_chunk_list_embedding,
                                                    rag_query=user_query,
                                                    graph=graph,
                                                    retriever=retriever,
                                                    recall_chunk_num=RECALL_CHUNK_NUM
                                                )
            response, usage = get_llm_response_via_api(
                                                        prompt=QUERY_PROMPT[DATASET].format_map({"question": user_query,
                                                        "materials": "\n\n".join(
                                                            retrieved_chunks)}),
                                                        API_KEY=os.getenv("API_KEY"),                                            
                                                        LLM_MODEL=LLM_MODEL,
                                                        TAU=TAU,
                                                        SEED=SEED,
                                                        )
            
            # Track response generation token usage
            if usage:
                total_response_input_tokens += getattr(usage, 'prompt_tokens', 0)
                total_response_output_tokens += getattr(usage, 'completion_tokens', 0)
            
            graph, dgl_graph, training_data = record_graph_construction(query=user_query,
                                                                        support_materials=retrieved_chunks,
                                                                        response=response, graph=graph,
                                                                        dgl_graph=dgl_graph,
                                                                        training_data=training_data,
                                                                        retriever=retriever,
                                                                        answer=user_answer[uid])
            print("{} Graph Construction: {}/{}".format(show_time(), uid + 1, len(user_question)))
            print(dgl_graph)
        # Save
        if TRAIN:
            store_nx(nx_obj=graph, path="./graph/{}_graph_{}.graphml".format(DATASET, ind))
            dgl.save_graphs(filename="./graph/{}_graph_{}.dgl".format(DATASET, ind), g_list=[dgl_graph])
            write_to_pkl(data=training_data, output_file="./graph/{}_training_data_{}.pkl".format(DATASET, ind))
        else:
            store_nx(nx_obj=graph, path="./graph/{}_test_graph_{}.graphml".format(DATASET, ind))
            dgl.save_graphs(filename="./graph/{}_test_graph_{}.dgl".format(DATASET, ind), g_list=[dgl_graph])
    
    # Print token usage summary
    total_input_tokens = total_query_gen_input_tokens + total_response_input_tokens
    total_output_tokens = total_query_gen_output_tokens + total_response_output_tokens
    total_tokens = total_input_tokens + total_output_tokens
    
    print("\n" + "="*60)
    print("TOKEN USAGE SUMMARY")
    print("="*60)
    print("Query Generation:")
    print(f"  Input tokens:  {total_query_gen_input_tokens:,}")
    print(f"  Output tokens: {total_query_gen_output_tokens:,}")
    print(f"  Total:         {total_query_gen_input_tokens + total_query_gen_output_tokens:,}")
    print("-"*60)
    print("Response Generation:")
    print(f"  Input tokens:  {total_response_input_tokens:,}")
    print(f"  Output tokens: {total_response_output_tokens:,}")
    print(f"  Total:         {total_response_input_tokens + total_response_output_tokens:,}")
    print("-"*60)
    print("Overall:")
    print(f"  Total input tokens:  {total_input_tokens:,}")
    print(f"  Total output tokens: {total_output_tokens:,}")
    print(f"  Total tokens:        {total_tokens:,}")
    print("="*60)