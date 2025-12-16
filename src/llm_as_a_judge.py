import json
import random
import dotenv
import os
import re
from collections import defaultdict

from src.llm import get_llm_response_via_api
from prompt.prompt import LLM_AS_JUDGE_PROMPT


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def parse_judge_json(text):
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty judge output")

    # Try strict JSON first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Fallback: extract first JSON object
    m = JSON_RE.search(text)
    if not m:
        raise ValueError(f"No JSON found in judge output:\n{text[:2000]}")
    return json.loads(m.group(0))


def make_prompt(template, query, gt, evidence, ans_a, ans_b):
    return (template
            .replace("{QUERY}", query or "")
            .replace("{GROUND_TRUTH}", gt or "")
            .replace("{EVIDENCE}", evidence or "")   # <-- FIXED
            .replace("{ANSWER_A}", ans_a or "")
            .replace("{ANSWER_B}", ans_b or ""))


def eval_pair(prompt_template, query, gt, evidence, a_name, a_text, b_name, b_text, llm_kwargs, seed=42):
    rng = random.Random(seed)
    pair = [(a_name, a_text), (b_name, b_text)]
    rng.shuffle(pair)
    nameA, textA = pair[0]
    nameB, textB = pair[1]

    prompt = make_prompt(prompt_template, query, gt, evidence, textA, textB)
    out, _ = get_llm_response_via_api(prompt, **llm_kwargs)

    # Try parse; if fail, return "skip"
    try:
        j = parse_judge_json(out)
    except Exception as e:
        return "skip", {
            "_presented_as": {"A": nameA, "B": nameB},
            "_error": str(e),
            "_raw_output_head": (out or "")[:2000],
        }

    # store mapping for audit/debug
    j["_presented_as"] = {"A": nameA, "B": nameB}

    w = str(j.get("overall_winner", "tie")).strip().upper()
    if w == "A": return nameA, j
    if w == "B": return nameB, j
    return "tie", j


def tournament(bm25, contriever, gor, sys_prompt, llm_kwargs, out_path="judge.jsonl", max_items=0):
    keys = sorted(set(bm25) & set(contriever) & set(gor))
    if max_items and max_items > 0:
        keys = keys[:max_items]

    stats = defaultdict(lambda: {"a_wins": 0, "b_wins": 0, "ties": 0, "skips": 0, "total": 0})

    with open(out_path, "w", encoding="utf-8") as f:
        for i, k in enumerate(keys):
            query = k.split(".", 1)[1] if "." in k else "Summarize the whole meeting."
            gt = bm25[k].get("ground_truth", "")
            evidence = ""  # optional

            for a, b in [("bm25", "contriever"), ("bm25", "gor"), ("contriever", "gor")]:
                a_text = {"bm25": bm25, "contriever": contriever, "gor": gor}[a][k].get("response", "")
                b_text = {"bm25": bm25, "contriever": contriever, "gor": gor}[b][k].get("response", "")

                winner, raw = eval_pair(sys_prompt, query, gt, evidence, a, a_text, b, b_text, llm_kwargs, seed=1000+i)

                rec = stats[(a, b)]
                rec["total"] += 1

                skipped = (winner == "skip")
                if skipped:
                    rec["skips"] += 1
                elif winner == "tie":
                    rec["ties"] += 1
                elif winner == a:
                    rec["a_wins"] += 1
                elif winner == b:
                    rec["b_wins"] += 1
                else:
                    rec["ties"] += 1

                f.write(json.dumps({
                    "key": k,
                    "pair": [a, b],
                    "winner": winner,
                    "skipped": skipped,
                    "presented_as": raw.get("_presented_as"),
                    "raw": raw
                }, ensure_ascii=False) + "\n")

    pairwise = {}
    for (a, b), rec in stats.items():
        non_tie = rec["a_wins"] + rec["b_wins"]
        if non_tie == 0:
            a_pct = b_pct = 0.0
        else:
            a_pct = 100.0 * rec["a_wins"] / non_tie
            b_pct = 100.0 * rec["b_wins"] / non_tie

        pairwise[f"{a}_vs_{b}"] = {
            a: round(a_pct, 1),
            b: round(b_pct, 1),
            "ties": rec["ties"],
            "skips": rec["skips"],
            "total": rec["total"],
            "non_tie_total": non_tie
        }

    return {"pairwise": pairwise, "out": out_path}


if __name__ == "__main__":
    bm25 = json.load(open("./result/qmsum_bm25_Meta-Llama-3.1-8B-Instruct-Turbo.json", "r", encoding="utf-8"))
    contriever = json.load(open("./result/qmsum_contriever_Meta-Llama-3.1-8B-Instruct-Turbo.json", "r", encoding="utf-8"))
    gor = json.load(open("./result/qmsum_gor.json", "r", encoding="utf-8"))

    dotenv.load_dotenv()

    llm_kwargs = dict(
        API_KEY=os.getenv("API_KEY"),
        LLM_MODEL="deepseek-ai/DeepSeek-R1",
        TAU=0.0
    )

    summary = tournament(bm25, contriever, gor, sys_prompt=LLM_AS_JUDGE_PROMPT, llm_kwargs=llm_kwargs)
    print(summary)
