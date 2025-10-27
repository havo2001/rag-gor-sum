import argparse
import os
# Really important, mac failed when import faises
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # must be BEFORE torch/faiss imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from dotenv import load_dotenv
from tqdm.auto import tqdm
from langchain_text_splitters import TokenTextSplitter

from src.helper import *
# from src.retrieval import *
from src.data_process import *
from src.llm import *
from prompt.prompt import QUERY_PROMPT_NORMAL
from src.bm25 import BM25
from src.contriever import Contriever


if __name__ == '__main__':
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--llm_model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default='mps')
    parser.add_argument("--tau", type=float, default=0)
    parser.add_argument("--retriever", type=str, default="contriever")
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--chunk_overlap", type=int, default=32)
    parser.add_argument("--recall_chunk_num", type=int, default=6) # The orginal paper takes 6 chunks


    opt = parser.parse_args()
    DATASET = opt.dataset
    LLM_MODEL = opt.llm_model
    SEED = opt.seed
    DEVICE = opt.device
    TAU = opt.tau
    RETRIEVER = opt.retriever
    CHUNK_SIZE = opt.chunk_size
    CHUNK_OVERLAP = opt.chunk_overlap
    RECALL_CHUNK_NUM = opt.recall_chunk_num
    

    set_seed(int(SEED))
    

    load_dotenv()

    # Get the retriever
    if RETRIEVER == 'contriever':
        retriever = Contriever(device=DEVICE)
    elif RETRIEVER == 'bm25':
        retriever = BM25()
    else:
        raise Exception("Dataset Error")


    # Initilize the langchain TokenTextSplitter with chunk_size and chunk_overlap the same in the original paper
    TEXT_SPLITTER = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # For the RAG with contriever only, we don't need to train anything so only call the test data for evaluation
    test_data = get_processed_data(DATASET, train=False)

    print("{} #Data: {}".format(show_time(), len(test_data)))
    test_data = test_data[:30]

    # Create the folder to store the result from rag
    if not os.path.exists("./result"):
        os.mkdir("./result")

    # Iterate to each sample
    result_recorder = dict()
    for idx, sample in enumerate(tqdm(test_data, total=len(test_data), desc="Evaluating", unit="ex")):
        all_doc_chunk_list = split_corpus_into_chunk(dataset=DATASET, sample=sample, text_splitter=TEXT_SPLITTER)
        test_gen = test_data_generation(dataset=DATASET, sample=sample)

        for test_query in test_gen:
            # Retrieve step, retrieve top 6 relevant chunks
            retrieved_idx, retrieved_chunks = retriever.retrieve(query=test_query['rag_query'],
                                                                 text_chunk=all_doc_chunk_list,
                                                                 chunk_num=RECALL_CHUNK_NUM)
        
            # Generate step
            prompt = QUERY_PROMPT_NORMAL[DATASET].format_map({'question': test_query['query'],
                                                             'materials': "\n\n".join(retrieved_chunks)})

            response = get_llm_response_via_api(prompt=prompt,
                                                API_KEY=os.getenv("API_KEY"),
                                                LLM_MODEL=LLM_MODEL,
                                                TAU=TAU,
                                                SEED=SEED)
            
            result_recorder[str(idx) + '.' + test_query['query']] = {"response": response, "ground_truth": test_query["summary"]}
    
    # Write the result into result folder
    llm_model_name = LLM_MODEL.split('/')[1] if '/' in LLM_MODEL else LLM_MODEL
    with open("./result/{}_{}_{}.json".format(DATASET, RETRIEVER, llm_model_name), 'w', encoding='utf-8') as file:
        json.dump(result_recorder, file, indent=4)




