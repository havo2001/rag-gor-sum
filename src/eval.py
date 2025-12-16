from collections import defaultdict
import numpy as np
import argparse
from rouge_score import rouge_scorer
import json
from src.helper import set_seed
from bert_score import score


def bert_score_eval(generate_response, ground_truth, device, batch_size=8):
    P, R, F = score(generate_response, ground_truth, model_type="microsoft/deberta-xlarge-mnli", device=device,
                    batch_size=batch_size)
    P = [float(i) for i in P.numpy()]
    R = [float(i) for i in R.numpy()]
    F = [float(i) for i in F.numpy()]

    return P, R, F


def rouge_eval(generated_response, ground_truth, type='rougeL'):
    scorer = rouge_scorer.RougeScorer([type], use_stemmer=True)
    # If the ground_truth is a list of possible answers
    if not isinstance(ground_truth, str):
        num_ref = len(ground_truth)
        generated_response_expand = generated_response * num_ref
        ground_truth_expand = ground_truth
        precisions = []
        recalls = []
        fmeasures = []
        for pred, target  in zip(generated_response_expand, ground_truth_expand):
            scores = scorer.score(prediction=pred, target=target)
            precisions.append(scores[type].precision)
            recalls.append(scores[type].recall)
            fmeasures.append(scores[type].fmeasure)
        precision, recall, fmeasure = max(precisions), max(recalls), max(fmeasures)
    else:
        scores = scorer.score(prediction=generated_response, target=ground_truth)
        precision = scores[type].precision
        recall = scores[type].recall
        fmeasure = scores[type].fmeasure

    return float(precision), float(recall), float(fmeasure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--retriever", type=str, default="contriever")
    parser.add_argument("--llm_model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--model_type", type=str, default="gor")
    
    opt = parser.parse_args()
    DATASET = opt.dataset
    RETRIEVER = opt.retriever
    LLM_MODEL = opt.llm_model
    MODEL_TYPE = opt.model_type

    llm_model_name = LLM_MODEL.split('/')[1] if '/' in LLM_MODEL else LLM_MODEL

    set_seed()

    if MODEL_TYPE == "gor":
        result_path = "./result/{}_gor.json".format(DATASET)
    elif MODEL_TYPE == "baseline":
        result_path = "./result/{}_{}_{}.json".format(DATASET, RETRIEVER, llm_model_name)
    else:
        raise Exception("Invalid model type")

    with open(result_path) as f:
        result_recorder = json.load(f)
   
    # For 3 metrics: Rouge-L, Rouge-1, Rouge-2. The original paper only cares about fmeasure
    metric = defaultdict(list)
    for key, val in result_recorder.items():
        pred = val['response']
        target = val['ground_truth']
        _, _, fL = rouge_eval(pred, target, type='rougeL')
        _, _, f1 = rouge_eval(pred, target, type='rouge1')
        _, _, f2 = rouge_eval(pred, target, type='rouge2')
        metric["ROUGE-L"].append(fL)
        metric["ROUGE-1"].append(f1)
        metric["ROUGE-2"].append(f2)

    final_metric = {key: np.mean(metric[key]) for key in metric.keys()}
    print(f"{'='*50} Result for RAG baseline with {RETRIEVER} {'='*50}")
    print(final_metric)


        




