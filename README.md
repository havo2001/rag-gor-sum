# RAG-GoR-Sum

A retrieval-augmented generation system for query-focused meeting summarization using the QMSum dataset.

## Overview

This project implements and compares different retrieval methods (BM25, Contriever, and Graph-of-Records) for query-focused summarization tasks. The system uses Meta-Llama-3.1-8B-Instruct-Turbo for text generation with greedy decoding (temperature=0) to ensure reproducible results.

## Installation

### Prerequisites
- Python 3.10+
- macOS/Linux (Windows may require additional configuration)

### Dependencies

Install the required packages:

```bash
# Python 3.10 recommended
pip install faiss-cpu          # Use faiss-cpu for macOS, faiss-gpu for Linux with CUDA
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
pip install networkx
pip install dgl==2.2.0
pip install bert_score
```

### Environment Setup

Create a `.env` file in the project root with your API key:

```
API_KEY=your_together_api_key_here
```

## Usage

### Running Baseline Experiments

#### BM25 Retrieval

Run the baseline with BM25:

```bash
python -m baseline.baseline --dataset 'qmsum' --retriever 'bm25' --llm_model 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'
```

Evaluate the results:

```bash
python -m src.eval --dataset 'qmsum' --retriever 'bm25' --llm_model 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo' --model_type baseline
```

#### Contriever Retrieval

Run the baseline with Contriever:

```bash
python -m baseline.baseline --dataset 'qmsum' --retriever 'contriever' --llm_model 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'
```

Evaluate the results:

```bash
python -m src.eval --dataset 'qmsum' --retriever 'contriever' --llm_model 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo' --model_type baseline
```

### Graph-of-Records (GoR)

The GoR implementation follows the official implementation from the paper.

#### Step 1: Graph Construction

Generate graphs for training set:

```bash
python -m GoR.graph_construction --dataset 'qmsum' --train
```

Generate graphs for test set:

```bash
python -m GoR.graph_construction --dataset 'qmsum'
```

#### Step 2: Training Preparation

```bash
python -m GoR.train_preparation --dataset 'qmsum'
```

#### Step 3: Training

```bash
python -m GoR.train --dataset 'qmsum'
```

#### Step 4: Inference on Test Set

```bash
python -m GoR.run --dataset 'qmsum' --llm_model 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'
```

#### Step 5: Evaluation

```bash
python -m src.eval --dataset 'qmsum' --llm_model 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo' --model_type gor
```

## Results

### ROUGE Scores on QMSum Dataset

| Method      | ROUGE-1              | ROUGE-2              | ROUGE-L              |
|-------------|----------------------|----------------------|----------------------|
| BM25        | 0.3253 (0.30, 0.35)  | 0.0894 (0.07, 0.11)  | 0.1884 (0.18, 0.20)  |
| Contriever  | 0.2977 (0.30, 0.34)  | 0.0751 (0.07, 0.09)  | 0.1709 (0.17, 0.19)  |
| GoR         | 0.3111 (0.29, 0.33)  | 0.0813 (0.07, 0.09)  | 0.1824 (0.17, 0.19)  |

*95% confidence intervals shown in parentheses*

### Token Usage Comparison

**Note:** GoR token usage includes tokens for both query generation and summarization.

| Method                                  | Total Input Tokens | Total Output Tokens |
|-----------------------------------------|-------------------:|--------------------:|
| Baseline (Avg of BM25 and Contriever)  | 46,403             | 9,761               |
| GoR                                     | 8,258,951          | 1,111,043           | 

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
├── baseline/                    # Baseline RAG implementations
│   └── baseline.py              # BM25 and Contriever baseline experiments
├── GoR/                         # Graph-of-Records implementation
│   ├── graph_construction.py   # Graph construction from documents
│   ├── train_preparation.py    # Prepare training data
│   ├── train.py                # Train GoR model
│   └── run.py                  # Inference on test set
├── src/                        # Core source code
│   ├── bm25.py                 # BM25 retrieval implementation
│   ├── contriever.py           # Contriever dense retrieval
│   ├── retrieval.py            # Retrieval utilities
│   ├── llm.py                  # LLM API interface
│   ├── eval.py                 # ROUGE evaluation scripts
│   ├── data_process.py         # Data preprocessing utilities
│   └── helper.py               # Helper functions
├── prompt/                     # Prompt templates
│   └── prompt.py               # Query and summarization prompts
├── data/                       # Dataset directory
│   └── raw/QMSum/              # QMSum dataset
├── graph/                      # Generated graphs (created during runtime)
├── weights/                    # Trained model weights (created during training)
├── result/                     # Experiment results (JSON files)
├── .env                        # API keys (create this file)
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```
