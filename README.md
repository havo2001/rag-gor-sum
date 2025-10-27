# rag-gor-sum
Some directions to explore:
- Do different methods give the different retrieved chunk or they are the same and the result was determined by the generate step.


`pip install faiss-gpu` this one doesn't work for mac, try this one `pip install faiss-cpu`
`pip install langchain_text_splitters`
`pip install langchain`
`pip install transformers`
`pip install torchvision`
`pip install torch`
`pip install rouge-score`
`pip install openai==0.28`
`pip install python-dotenv` for loading API key
`pip install rank_bm25`
`pip install tiktoken`

## CLI:
Create the summary
`python3 - m baseline.baseline --dataset 'qmsum' --retriever 'bm25' --llm 'mistralai Mixtral-8x7B-Instruct-v0.1'` Together AI doesn't support serverless for LLaMA-2-7b-chat. (maybe use another service)

Get the rouge score for baseline:



Got the problem with OpenMP: 
This helped fix temporary but should figure out how to fix it later!!!
`os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # must be BEFORE torch/faiss imports`
`os.environ["TOKENIZERS_PARALLELISM"] = "false"`
`os.environ.setdefault("OMP_NUM_THREADS", "1")`
`os.environ.setdefault("MKL_NUM_THREADS", "1")`



### BM25:
`python3 -m baseline.baseline --dataset 'qmsum' --retriever 'bm25' --llm  'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'`

Evaluate:
`python3 -m src.eval --dataset 'qmsum' --retriever 'bm25' --llm  'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'`

`{'ROUGE-L': np.float64(0.1872924315967513), 'ROUGE-1': np.float64(0.32021873361282305), 'ROUGE-2': np.float64(0.09003305137058028)}`


### Contriever:
`python3 -m baseline.baseline --dataset 'qmsum' --retriever 'contriever' --llm  'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'`

Evaluate:
`python3 -m src.eval --dataset 'qmsum' --retriever 'contriever' --llm  'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'`

`{'ROUGE-L': np.float64(0.1708931477428153), 'ROUGE-1': np.float64(0.2976901598818958), 'ROUGE-2': np.float64(0.07512939069385761)}`


