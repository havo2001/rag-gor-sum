import json
import re
from collections import defaultdict

from src.eval import rouge_eval



def pick_top_rouge():
    # Function to pick the test examples that GoR performs better than BM25 with ROUGE score

    result_gor = "./result/qmsum_gor.json"
    result_bm25 = "./result/qmsum_bm25_Meta-Llama-3.1-8B-Instruct-Turbo.json"
   

    with open(result_gor) as f:
        result_recorder_gor = json.load(f)

    with open(result_bm25) as f:
        result_recorder_bm25 = json.load(f)
   
    # For 3 metrics: Rouge-L, Rouge-1, Rouge-2. The original paper only cares about fmeasure
    metric_gor = defaultdict(list)
    for key, val in result_recorder_gor.items():
        pred = val['response']
        target = val['ground_truth']
        _, _, fL = rouge_eval(pred, target, type='rougeL')
        _, _, f1 = rouge_eval(pred, target, type='rouge1')
        _, _, f2 = rouge_eval(pred, target, type='rouge2')
        metric_gor["ROUGE-L"].append(fL)
        metric_gor["ROUGE-1"].append(f1)
        metric_gor["ROUGE-2"].append(f2)
    
    metric_bm25 = defaultdict(list)
    for key, val in result_recorder_bm25.items():
        pred = val['response']
        target = val['ground_truth']
        _, _, fL = rouge_eval(pred, target, type='rougeL')
        _, _, f1 = rouge_eval(pred, target, type='rouge1')
        _, _, f2 = rouge_eval(pred, target, type='rouge2')
        metric_bm25["ROUGE-L"].append(fL)
        metric_bm25["ROUGE-1"].append(f1)
        metric_bm25["ROUGE-2"].append(f2)

    
    result = []
    print("Test examples that GoR performs better than BM25 with ROUGE score:")
    for i in range(len(result_recorder_gor)):
        if metric_gor["ROUGE-L"][i] > metric_bm25["ROUGE-L"][i] and metric_gor["ROUGE-1"][i] > metric_bm25["ROUGE-1"][i] and metric_gor["ROUGE-2"][i] > metric_bm25["ROUGE-2"][i]:
            result.append(i)
    return result  


def pick_top_LLM_as_a_judge():
    keys = []
    idxs = []
    IDX_RE = re.compile(r"^(\d+)\.")

    with open("judge.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)

            if j.get("skipped", False):
                continue

            pair = j.get("pair", [])
            if set(pair) != {"bm25", "gor"}:
                continue

            if j.get("winner") != "gor":
                continue

            k = j.get("key", "")
            keys.append(k)

            m = IDX_RE.match(k)
            if m:
                idxs.append(int(m.group(1)))

    return idxs


if __name__ == "__main__":
    rouge_best_idx = pick_top_rouge()
    llm_as_judge_best_idx = pick_top_LLM_as_a_judge()

    # Get the ground truth and the response from the best idx
    with open("./result/qmsum_gor.json", "r", encoding="utf-8") as f:
        result_recorder_gor = json.load(f)
        result_recorder_gor_keys = list(result_recorder_gor.keys())
    with open("./result/qmsum_bm25_Meta-Llama-3.1-8B-Instruct-Turbo.json", "r", encoding="utf-8") as f:
        result_recorder_bm25 = json.load(f)
        result_recorder_bm25_keys = list(result_recorder_bm25.keys())

    # put this near the top of your main (before the loops)
    out_path = "./result/error_analysis_dump.txt"
    out = open(out_path, "w", encoding="utf-8")

    # Replace each print(...) with: print(..., file=out)
    for idx in rouge_best_idx:
        print(f"Rouge best idx: {idx}", file=out)
        print("Ground truth:", file=out)
        print(result_recorder_gor[result_recorder_gor_keys[idx]]['ground_truth'], file=out)
        print("--------------------------------", file=out)
        print("GoR response:", file=out)
        print(result_recorder_gor[result_recorder_gor_keys[idx]]['response'], file=out)
        print("--------------------------------", file=out)
        print("BM25 response:", file=out)
        print(result_recorder_bm25[result_recorder_bm25_keys[idx]]['response'], file=out)
        print("****************************************************", file=out)

    for idx in llm_as_judge_best_idx:
        print(f"LLM as judge best idx: {idx}", file=out)
        print("Ground truth:", file=out)
        print(result_recorder_gor[result_recorder_gor_keys[idx]]['ground_truth'], file=out)
        print("--------------------------------", file=out)
        print("GoR response:", file=out)
        print(result_recorder_gor[result_recorder_gor_keys[idx]]['response'], file=out)
        print("--------------------------------", file=out)
        print("BM25 response:", file=out)
        print(result_recorder_bm25[result_recorder_bm25_keys[idx]]['response'], file=out)
        print("****************************************************", file=out)

    out.close()
    print(f"Saved to: {out_path}")

