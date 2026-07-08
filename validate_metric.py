#!/usr/bin/env python3
"""Validate that the composite accuracy metric reflects real reply quality.

For each test email, scores four candidate replies through the exact same
evaluation pipeline defined in evaluate.py (imported, not duplicated):

  1. reference    — the real sent_reply for that email        (expected: high)
  2. generic       — polite, no resolution, boilerplate         (expected: mid-low)
  3. wrong_topic   — a genuine reply, but for a different case  (expected: low relevance)
  4. rude          — curt and dismissive                        (expected: low tone)

If reference > generic > wrong_topic and reference > rude hold on mean
composite score across all test emails, the metric is discriminating real
quality differences rather than producing arbitrary numbers.
"""

import json
import os
import sys
from pathlib import Path
from statistics import mean

from google import genai

from evaluate import DATASET_PATH, evaluate_response
from generate_replies import build_retrieval_index, retrieve_top_k

BASE_DIR = Path(__file__).resolve().parent
VALIDATION_PATH = BASE_DIR / "outputs" / "validation.json"

# This script needs 24 judge calls (6 test emails x 4 classes) in one run,
# which exceeds gemini-2.5-flash's free-tier daily cap (20 req/day). The
# lite tier has a materially higher free daily cap, so judge calls here use
# it explicitly rather than evaluate.py's default MODEL.
JUDGE_MODEL = "gemini-2.5-flash-lite"

GENERIC_REPLY = "Thanks for reaching out. We'll look into this and get back to you. — Northwind Gear Support"
RUDE_REPLY = "We already told you the answer. Read the FAQ next time. — Northwind Gear Support"

CLASSES = ["reference", "generic", "wrong_topic", "rude"]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_candidates(test_pairs):
    """Build the 4 candidate replies for each test email.

    wrong_topic uses the sent_reply of the next test email in the list. Since
    this dataset has exactly one test email per category, rotating by one
    always lands on a genuinely different category — asserted below rather
    than assumed.
    """
    n = len(test_pairs)
    candidates = {}
    for i, pair in enumerate(test_pairs):
        wrong_topic_pair = test_pairs[(i + 1) % n]
        assert wrong_topic_pair["category"] != pair["category"], (
            "wrong_topic candidate must come from a different category"
        )
        candidates[pair["id"]] = {
            "reference": pair["sent_reply"],
            "generic": GENERIC_REPLY,
            "wrong_topic": wrong_topic_pair["sent_reply"],
            "rude": RUDE_REPLY,
        }
    return candidates


def score_all_candidates(client, test_pairs, id_to_pair, train_pairs):
    candidates_by_id = build_candidates(test_pairs)

    print("Building TF-IDF retrieval index (same as generate_replies.py) ...")
    vectorizer, matrix = build_retrieval_index(train_pairs)

    results = []
    total = len(test_pairs) * len(CLASSES)
    done = 0
    for pair in test_pairs:
        # Retrieval depends only on the incoming email, so it's computed once
        # per test email and shared across all 4 candidate classes — keeps
        # the grounding context identical, so score differences are
        # attributable to the candidate reply, not to different examples.
        examples = retrieve_top_k(vectorizer, matrix, train_pairs, pair["incoming_email"])
        retrieved_ids = [p["id"] for p, _score in examples]

        for cls in CLASSES:
            done += 1
            reply_text = candidates_by_id[pair["id"]][cls]
            print(f"[{done}/{total}] test id={pair['id']} category={pair['category']} class={cls} ...")

            entry = {
                "id": pair["id"],
                "category": pair["category"],
                "incoming_email": pair["incoming_email"],
                "generated_reply": reply_text,
                "retrieved_example_ids": retrieved_ids,
            }
            result = evaluate_response(client, entry, id_to_pair, model=JUDGE_MODEL)
            result["class"] = cls

            if result.get("status") == "ok":
                print(f"  composite={result['composite_score']}")
            else:
                print(f"  {result.get('status')}: {result.get('judge_error')}")

            results.append(result)

    return results


def summarize_by_class(results):
    scores_by_class = {cls: [] for cls in CLASSES}
    for r in results:
        if r.get("status") == "ok":
            scores_by_class[r["class"]].append(r["composite_score"])

    per_class_mean = {
        cls: (round(mean(scores), 2) if scores else None)
        for cls, scores in scores_by_class.items()
    }
    return per_class_mean


def check_ordering(per_class_mean):
    ref, gen, wrong, rude = (
        per_class_mean["reference"],
        per_class_mean["generic"],
        per_class_mean["wrong_topic"],
        per_class_mean["rude"],
    )

    have_all_ref_gen_wrong = None not in (ref, gen, wrong)
    ordering_ref_gen_wrong = bool(have_all_ref_gen_wrong and ref > gen > wrong)

    have_ref_rude = None not in (ref, rude)
    ordering_ref_rude = bool(have_ref_rude and ref > rude)

    separations = {
        "reference_minus_generic": round(ref - gen, 2) if None not in (ref, gen) else None,
        "generic_minus_wrong_topic": round(gen - wrong, 2) if None not in (gen, wrong) else None,
        "reference_minus_wrong_topic": round(ref - wrong, 2) if None not in (ref, wrong) else None,
        "reference_minus_rude": round(ref - rude, 2) if None not in (ref, rude) else None,
    }

    return ordering_ref_gen_wrong, ordering_ref_rude, separations


def print_report(per_class_mean, ordering_ref_gen_wrong, ordering_ref_rude, separations):
    print()
    print("=" * 70)
    print("PER-CLASS MEAN COMPOSITE SCORE")
    print("=" * 70)
    for cls in CLASSES:
        val = per_class_mean[cls]
        print(f"  {cls:<14} {val if val is not None else 'N/A'}")

    print()
    print("Score separation:")
    for k, v in separations.items():
        print(f"  {k:<32} {v if v is not None else 'N/A'}")

    print()
    print(f"Ordering reference > generic > wrong_topic: {ordering_ref_gen_wrong}")
    print(f"Ordering reference > rude: {ordering_ref_rude}")
    print()
    if ordering_ref_gen_wrong and ordering_ref_rude:
        print("METRIC VALIDATED: discriminates quality classes correctly.")
    else:
        print("METRIC VALIDATION FAILED: expected ordering does not hold.")


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset from {DATASET_PATH} ...")
    dataset = load_json(DATASET_PATH)
    id_to_pair = {d["id"]: d for d in dataset}
    train_pairs = [d for d in dataset if d["split"] == "train"]
    test_pairs = [d for d in dataset if d["split"] == "test"]
    print(f"{len(train_pairs)} train pairs, {len(test_pairs)} test emails — "
          f"{len(test_pairs) * len(CLASSES)} total (email, candidate class) pairs to score.")

    client = genai.Client(api_key=api_key)

    results = score_all_candidates(client, test_pairs, id_to_pair, train_pairs)
    per_class_mean = summarize_by_class(results)
    ordering_ref_gen_wrong, ordering_ref_rude, separations = check_ordering(per_class_mean)
    metric_validated = ordering_ref_gen_wrong and ordering_ref_rude

    validation = {
        "per_class_mean_composite": per_class_mean,
        "ordering_reference_gt_generic_gt_wrong_topic": ordering_ref_gen_wrong,
        "ordering_reference_gt_rude": ordering_ref_rude,
        "metric_validated": metric_validated,
        "score_separation": separations,
        "responses": results,
    }

    VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_PATH, "w") as f:
        json.dump(validation, f, indent=2)

    print_report(per_class_mean, ordering_ref_gen_wrong, ordering_ref_rude, separations)
    print(f"\nSaved full breakdown to {VALIDATION_PATH}")


if __name__ == "__main__":
    main()
