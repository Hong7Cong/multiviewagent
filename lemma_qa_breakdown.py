"""Per-question-type accuracy breakdown of LEMMA QA results across view conditions.

Reads outputs/qa_all_*.jsonl (one file per view condition, written by qwen3vl_qa.py /
qwen3vl_qa_multiview.py) and writes a markdown table to results/LEMMA.md.

    python lemma_qa_breakdown.py
    python lemma_qa_breakdown.py --pattern "outputs/qa_all_*_8b.jsonl" --output results/LEMMA.md
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

QTYPES = [f"Q{i}" for i in range(1, 9)]

# Short descriptions inferred from the questions in LEMMA/subsets/2x1_2x2_qa.json.
QTYPE_DESC = {
    "Q1": "Initial object location",
    "Q2": "Person location during action",
    "Q3": "Object destination after action",
    "Q4": "Egocentric relative direction",
    "Q5": "Object state change",
    "Q6": "Object-object spatial relation",
    "Q7": "Other person's location (multi-agent)",
    "Q8": "Activity order",
}

# Display order of view conditions: single views first, then pairs.
VIEW_ORDER = ["fpv1", "fpv2", "master", "fpv1+fpv2", "fpv1+master", "fpv2+master"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", default="outputs/qa_all_*.jsonl", help="glob for results files")
    p.add_argument("--output", default="results/LEMMA.md", help="markdown file to write")
    return p.parse_args()


def qtype_of(qid):
    """'26l_21_5_1_Q8_002' -> 'Q8'"""
    return qid.split("_")[-2]


def condition_of(path):
    """'outputs/qa_all_fpv1+master_8b.jsonl' -> ('fpv1+master', '8b')"""
    m = re.match(r"qa_all_(.+)_([^_]+)\.jsonl$", os.path.basename(path))
    return m.group(1), m.group(2)


def load(path):
    """Returns {qtype: [n_correct, n_total]}, plus the 'All' row."""
    counts = defaultdict(lambda: [0, 0])
    with open(path) as f:
        for r in map(json.loads, f):
            for key in (qtype_of(r["id"]), "All"):
                counts[key][0] += bool(r["correct"])
                counts[key][1] += 1
    return counts


def sort_key(cond):
    view, model = cond
    return (model, VIEW_ORDER.index(view) if view in VIEW_ORDER else len(VIEW_ORDER), view)


def fmt(c, n, best):
    if not n:
        return "-"
    s = f"{100 * c / n:.1f}"
    return f"**{s}**" if best else s


def main():
    args = parse_args()
    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f"no files match {args.pattern}")
    results = {condition_of(p): load(p) for p in paths}
    conds = sorted(results, key=sort_key)

    cols = QTYPES + ["All"]
    # Question counts per type (sanity check that every condition answered the same set).
    ns = {cond: [results[cond][q][1] for q in cols] for cond in conds}
    ref_n = ns[conds[0]]
    mismatched = [f"{v}_{m}" for (v, m), n in ns.items() if n != ref_n]

    def acc(cond, q):
        c, n = results[cond][q]
        return c / n if n else -1

    best = {q: max(acc(c, q) for c in conds) for q in cols}

    lines = [
        "# LEMMA QA results (2x1 + 2x2 takes)",
        "",
        f"**Table 1.** Accuracy (%) of Qwen3-VL per question type (Q1-Q8, see Table 2) under each "
        f"view condition on LEMMA 2x1 + 2x2 takes. The first row gives the number of questions per type. "
        f"Best per column in bold. Source: `{args.pattern}`.",
        "",
        "| Views | Model | " + " | ".join(cols) + " |",
        "|---|---|" + "---:|" * len(cols),
        "| *#Questions* | | " + " | ".join(str(n) for n in ref_n) + " |",
    ]
    for cond in conds:
        view, model = cond
        cells = [fmt(*results[cond][q], best=abs(acc(cond, q) - best[q]) < 1e-12) for q in cols]
        lines.append(f"| {view} | {model} | " + " | ".join(cells) + " |")
    if mismatched:
        lines += ["", f"> Warning: question counts differ for: {', '.join(mismatched)}"]

    lines += [
        "",
        "## Question types",
        "",
        "**Table 2.** LEMMA question types and the number of questions of each type.",
        "",
        "| Type | Description | #Questions |", "|---|---|---:|"]
    for q, n in zip(QTYPES, ref_n):
        lines.append(f"| {q} | {QTYPE_DESC.get(q, '')} | {n} |")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
