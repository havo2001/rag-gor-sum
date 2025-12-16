import argparse

import dgl
from tqdm import tqdm
from langchain_text_splitters import TokenTextSplitter

from src.contriever import Contriever
from src.helper import *
from src.eval import bert_score_eval
from src.data_process import get_processed_data, split_corpus_into_chunk


def training_data_generation(graph, retriever,training_data, device):
    queries = [i["query"] for i in training_data]
    queries_embedding = retriever.get_dense_embedding(
        instructions=queries,
        tokenizer=retriever.query_tokenizer,
        model=retriever.query_encoder
    )
    queries_embedding = [i.cpu() for i in queries_embedding]
    bert_score = None
    if "answer" in training_data[0]:
        responses = []
        for node, attrs in graph.nodes(data=True):
            responses.append(node)
        answers = []
        for i in training_data:
            answers.extend([i["answer"]] * len(responses))
        responses = responses * len(training_data)
        _, _, bert_score = bert_score_eval(generate_response=responses, ground_truth=answers, device=device)
        bert_score = np.array(bert_score).reshape((len(training_data), -1))
        # print(bert_score.shape)

    return queries_embedding, bert_score


def integrate_isolated(graph, dgl_graph, all_doc_chunk_list, all_doc_chunk_list_embedding):
    raw_chunk = []
    for node, attrs in graph.nodes(data=True):
        raw_chunk.append(node)
    non_dup_chunk = []
    non_dup_chunk_embedding = []
    for chunk, chunk_embedding in zip(all_doc_chunk_list, all_doc_chunk_list_embedding):
        if chunk not in raw_chunk:
            graph.add_node(chunk)
            raw_chunk.append(chunk)
            non_dup_chunk.append(chunk)
            non_dup_chunk_embedding.append(chunk_embedding)

    if len(non_dup_chunk) != 0:
        dgl_graph.add_nodes(num=len(non_dup_chunk), data={'feat': torch.vstack(non_dup_chunk_embedding).cpu()})

    return graph, dgl_graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--retriever", type=str, default="contriever")
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--chunk_overlap", type=int, default=32)
    parser.add_argument("--recall_chunk_num", type=int, default=6)
    opt = parser.parse_args()
    DATASET = opt.dataset
    SEED = opt.seed
    DEVICE = opt.device
    RETRIEVER = opt.retriever
    CHUNK_SIZE = opt.chunk_size
    CHUNK_OVERLAP = opt.chunk_overlap
    RECALL_CHUNK_NUM = opt.recall_chunk_num

    set_seed(int(SEED))
    
    retriever = Contriever(device=DEVICE)

    TEXT_SPLITTER = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    data = get_processed_data(dataset=DATASET, train=True)
    print("{} #Data: {}".format(show_time(), len(data)))
    data = data[:400]
    query_embedding_list = []
    bert_score_list = []
    gs_list = []
    for ind, sample in tqdm(enumerate(data), total=len(data)):
        all_doc_chunk_list = split_corpus_into_chunk(dataset=DATASET, sample=sample, text_splitter=TEXT_SPLITTER)
        all_doc_chunk_list_embedding = retriever.get_dense_embedding(
            instructions=all_doc_chunk_list, 
            tokenizer=retriever.ctx_tokenizer, 
            model=retriever.ctx_encoder)
        
        try:
            graph = load_nx(path="./graph/{}_graph_{}.graphml".format(DATASET, ind))
            gs, _ = dgl.load_graphs("./graph/{}_graph_{}.dgl".format(DATASET, ind))
            dgl_graph = gs[0]
            training_data = read_from_pkl(output_file="./graph/{}_training_data_{}.pkl".format(DATASET, ind))
        except Exception as e:
            print(e)
            continue
        graph, dgl_graph = integrate_isolated(graph=graph, dgl_graph=dgl_graph, all_doc_chunk_list=all_doc_chunk_list,
                                              all_doc_chunk_list_embedding=all_doc_chunk_list_embedding)
        queries_embedding, bert_score = training_data_generation(graph=graph, retriever=retriever, training_data=training_data, device=DEVICE)
        gs_list.append(dgl_graph)
        query_embedding_list.append(queries_embedding)
        bert_score_list.append(bert_score)

    check_path("./training_data")
    dgl.save_graphs("./training_data/{}_gs.dgl".format(DATASET), gs_list)
    write_to_pkl(data=query_embedding_list, output_file="./training_data/{}_qe.pkl".format(DATASET))
    write_to_pkl(data=bert_score_list, output_file="./training_data/{}_bs.pkl".format(DATASET))
