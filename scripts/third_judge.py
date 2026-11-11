r"""
Third, independent judge from a DIFFERENT model family (e.g. Google Gemini with the three PDFs attached).

  python scripts/third_judge.py export            # writes paste-ready batches of 10 answers + a CSV template
  python scripts/third_judge.py import gemini_verdicts.csv --model "<exact Gemini model name>"

EXPORT uses the SAME fixed, category-stratified sample of 100 answers as the second-judge cross-check, so the paper can
report agreement between three judges on identical items. The files go to
    <workspace>/benchmark/third_judge/batch_01.md ... batch_10.md   (paste one at a time into Gemini, PDFs attached)
    <workspace>/benchmark/third_judge/verdicts_template.csv          (id, category, question, ..., verdict)
Gemini only sees the question, its type and the audited answer (plus the PDFs you attach): it is NOT shown the other
judges' verdicts or the gold reference, so its labels are independent.

IMPORT accepts any CSV/text with lines "B0123, SUPPORTED" (or a 'verdict' column), computes Cohen's kappa against the
primary judge and every other stored judge, and writes <workspace>/benchmark/judge_third.json (read by the paper
exporter).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from kg_audit.utils import workspace as wsm  # noqa: E402
from kg_audit.utils.main_utils import read_json, read_jsonl, write_json_atomic  # noqa: E402

VERDICTS = ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED")
EXPECT = {"FACTUAL": "answerable from the papers", "CROSS_DOC": "answerable by combining two of the papers",
          "BORDERLINE": "NOT answerable from the papers; a faithful answer says so",
          "FALSE_PREMISE": "contains a false premise; a faithful answer corrects it"}
INSTRUCTIONS = """You are a meticulous scientific fact-checking judge. Three PDFs are attached: DeepWalk (Perozzi et al.,
2014), LINE (Tang et al., 2015) and node2vec (Grover and Leskovec, 2016). For EACH item below, decide whether the AI
ANSWER is faithful to these papers.

Rules
- A claim is SUPPORTED if the papers state or directly entail it, or if it is uncontroversial textbook background.
- It is UNSUPPORTED if it attributes to the papers something they do not contain or contradict.
- If the question contains a false premise and the answer goes along with it, that is UNSUPPORTED.
- If the question is not answerable from the papers, inventing specifics is UNSUPPORTED; saying the papers do not
  report it is SUPPORTED.
Verdict per item: SUPPORTED (no unsupported claim), PARTIALLY_SUPPORTED (some unsupported claims) or UNSUPPORTED
(the core of the answer is unsupported).

Reply ONLY with one line per item, exactly in this form and nothing else:
ID,VERDICT
(for example: B0042,PARTIALLY_SUPPORTED)
"""


def paths():
    W = wsm.workspace_dir(wsm.active_workspace_id())
    B = os.path.join(W, "benchmark")
    return W, B, os.path.join(B, "runs", "latest"), os.path.join(B, "third_judge")


def export(per_batch: int = 10) -> None:
    W, B, R, out = paths()
    ids = (read_json(os.path.join(R, "crosscheck_ids.json"), default={}) or {}).get("ids") or []
    if not ids:
        sys.exit("No cross-check sample found - run the benchmark first.")
    items = {it["id"]: it for it in read_jsonl(os.path.join(B, "benchmark.jsonl"))}
    os.makedirs(out, exist_ok=True)
    rows = []
    for i in ids:
        ans = (read_json(os.path.join(R, "intercepts", f"{i}.json"), default={}) or {}).get("answer", "")
        it = items[i]
        rows.append({"id": i, "category": it.get("category", ""), "question": it["question"], "answer": ans.strip()})
    for b in range(0, len(rows), per_batch):
        chunk = rows[b:b + per_batch]
        text = INSTRUCTIONS + "\n" + "\n".join(
            f"=== ITEM {r['id']} ===\nQUESTION TYPE: {r['category']} - {EXPECT.get(r['category'], '')}\nQUESTION: {r['question']}\n"
            f"AI ANSWER:\n{r['answer']}\n" for r in chunk)
        with open(os.path.join(out, f"batch_{b // per_batch + 1:02d}.md"), "w", encoding="utf-8") as f:
            f.write(text)
    with open(os.path.join(out, "verdicts_template.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "category", "question", "answer", "verdict"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "verdict": ""})
    print(f"Wrote {len(rows)} items in {(len(rows) + per_batch - 1) // per_batch} batches to {out}\n"
          f"Paste each batch_XX.md into Gemini (with the three PDFs attached), collect the 'ID,VERDICT' lines into one "
          f"file, then run:  python scripts/third_judge.py import <that file> --model \"<Gemini model name>\"")


def _parse(path: str):
    out = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    for m in re.finditer(r"\b(B\d{4})\b[^A-Za-z\n]*\b(PARTIALLY[_ ]SUPPORTED|UNSUPPORTED|SUPPORTED)\b", text, re.I):
        out[m.group(1)] = m.group(2).upper().replace(" ", "_")
    return out


def do_import(path: str, model: str) -> None:
    from sklearn.metrics import cohen_kappa_score
    W, B, R, out = paths()
    third = _parse(path)
    if not third:
        sys.exit("No 'B0123,VERDICT' lines found in that file.")
    judges = {"primary": {f[:-5]: (read_json(os.path.join(R, "judgements", f)) or {}) for f in os.listdir(os.path.join(R, "judgements"))}}
    for d in (os.listdir(os.path.join(R, "crosscheck")) if os.path.isdir(os.path.join(R, "crosscheck")) else []):
        judges[d] = {f[:-5]: read_json(os.path.join(R, "crosscheck", d, f)) or {} for f in os.listdir(os.path.join(R, "crosscheck", d))}
    for d in (os.listdir(os.path.join(R, "judgements_archive")) if os.path.isdir(os.path.join(R, "judgements_archive")) else []):
        judges[d] = {f[:-5]: read_json(os.path.join(R, "judgements_archive", d, f)) or {} for f in os.listdir(os.path.join(R, "judgements_archive", d))}
    primary_model = next((j.get("judge_model") for j in judges["primary"].values() if j.get("judge_model")), "primary")
    res = {"third_judge": model, "n": len(third), "primary_judge": primary_model, "agreement": {}}
    for name, lab in judges.items():
        ids = [i for i in third if lab.get(i, {}).get("verdict")]
        if len(ids) < 10:
            continue
        a = [lab[i]["verdict"] for i in ids]
        b = [third[i] for i in ids]
        ya, yb = [int(x == "SUPPORTED") for x in a], [int(x == "SUPPORTED") for x in b]
        key = primary_model if name == "primary" else name
        res["agreement"][key] = {"n": len(ids), "kappa_binary": float(cohen_kappa_score(ya, yb)),
                                 "raw_agreement_binary": float(sum(x == y for x, y in zip(ya, yb)) / len(ids)),
                                 "kappa_3class": float(cohen_kappa_score(a, b))}
    res["supported_rate_third"] = sum(v == "SUPPORTED" for v in third.values()) / len(third)
    res["verdicts"] = third
    write_json_atomic(os.path.join(B, "judge_third.json"), res)
    print(json.dumps({k: v for k, v in res.items() if k != "verdicts"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "import"])
    ap.add_argument("file", nargs="?")
    ap.add_argument("--model", default="gemini")
    a = ap.parse_args()
    export() if a.cmd == "export" else do_import(a.file, a.model)
