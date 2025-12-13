# RAG-GoR-Sum

A retrieval-augmented generation system for query-focused meeting summarization using the QMSum dataset.

## Overview

This project implements and compares different retrieval methods (BM25 and Contriever) for query-focused summarization tasks. The system uses Meta-Llama-3.1-8B-Instruct-Turbo for text generation with greedy decoding (temperature=0) to ensure reproducible results.

## Installation

### Prerequisites
- Python 3.8+
- macOS/Linux (Windows may require additional configuration)

### Dependencies

Install the required packages:

```bash
pip install faiss-cpu  # Use faiss-cpu for macOS, faiss-gpu for Linux with CUDA
pip install torch
pip install torchvision
pip install transformers
pip install langchain
pip install langchain_text_splitters
pip install rouge-score
pip install openai==0.28
pip install python-dotenv
pip install rank_bm25
pip install tiktoken
```

### Environment Setup

Create a `.env` file in the project root with your API key:

```
TOGETHER_API_KEY=your_api_key_here
```

## Usage

### Running Baseline Experiments

#### BM25 Retrieval

Run the baseline with BM25:

```bash
python3 -m baseline.baseline --dataset 'qmsum' --retriever 'bm25' --llm 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'
```

Evaluate the results:

```bash
python3 -m src.eval --dataset 'qmsum' --retriever 'bm25' --llm 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'
```

#### Contriever Retrieval

Run the baseline with Contriever:

```bash
python3 -m baseline.baseline --dataset 'qmsum' --retriever 'contriever' --llm 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'
```

Evaluate the results:

```bash
python3 -m src.eval --dataset 'qmsum' --retriever 'contriever' --llm 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'
```

## Results

Evaluation metrics on QMSum dataset:

| Method      | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------------|---------|---------|---------|
| BM25        | 0.3202  | 0.0900  | 0.1873  |
| Contriever  | 0.2977  | 0.0751  | 0.1709  |

## Known Issues

### OpenMP Library Conflict

If you encounter OpenMP errors on macOS, add these environment variables at the top of your script (before importing torch/faiss):

```python
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
```

## Research Directions

- Investigate whether different retrieval methods retrieve distinct chunks or if performance differences stem primarily from the generation step
- Explore alternative LLM services (Together AI has limited serverless support for some models)

## Project Structure

```
rag-gor-sum/
├── baseline/          # Baseline RAG implementations
├── src/              # Source code
│   ├── bm25.py       # BM25 retrieval implementation
│   ├── contriever.py # Contriever retrieval implementation
│   ├── retrieval.py  # Retrieval utilities
│   ├── llm.py        # LLM interface
│   └── eval.py       # Evaluation scripts
├── prompt/           # Prompt templates
├── data/             # Dataset directory
└── result/           # Experiment results


