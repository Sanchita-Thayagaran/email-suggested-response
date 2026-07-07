#!/usr/bin/env python3
"""Accuracy system for generated support-reply candidates.

Three complementary layers per response:
  1. LLM-as-judge rubric (primary signal) — 5 dimensions, 1-5 each.
  2. Deterministic checks (pass/fail) — structural/format sanity checks.
  3. Reference similarity (secondary signal only) — TF-IDF cosine similarity
     against the held-out sent_reply. Never gates the score; many different
     replies can be correct, so this is reported but weighted lightly.

Composite per-response score (0-100):
    70% judge   (mean of the 5 dimension scores, linearly scaled 1-5 -> 0-100)
  + 20% deterministic pass rate (checks passed / checks applicable, -> 0-100)
  + 10% reference similarity (cosine similarity, -> 0-100)
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean

from google import genai
from google.genai import errors, types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.json"
GENERATED_PATH = BASE_DIR / "outputs" / "generated.json"
RESULTS_PATH = BASE_DIR / "outputs" / "results.json"

MODEL = "gemini-2.5-flash"
MAX_TOKENS = 1024
TEMPERATURE = 0

MAX_RETRIES = 2
RETRY_SLEEP_SECONDS = 2

JUDGE_DIMENSIONS = ["relevance", "grounding", "tone", "actionability", "consistency"]

ORDER_NUMBER_RE = re.compile(r"#NW-\d{5}")
GREETING_RE = re.compile(r"^\s*(hi|hello|hey|dear|greetings)\b", re.IGNORECASE)
SIGNOFF_RE = re.compile(r"Northwind Gear Support\s*$", re.IGNORECASE)
PLACEHOLDER_RE = re.compile(r"\[[A-Za-z0-9 _]+\]|\{\{|\btodo\b", re.IGNORECASE)

WORD_COUNT_MIN = 40
WORD_COUNT_MAX = 250

JUDGE_SYSTEM_PROMPT = """You are a strict, impartial quality judge for customer support email replies at Northwind Gear, an outdoor-gear e-commerce company.

Company policies (ground truth — the reply should never contradict these or invent others):
- Returns: 30-day return window from the delivery date, full refund to the original payment method.
- Warranty: 2-year warranty on backpacks ("packs") against manufacturing defects. No other product category carries a stated warranty.
- Shipping: standard shipping takes 5-7 business days.

You will be given:
1. The incoming customer email.
2. Past email/reply pairs that were retrieved and given to the reply-writing model as grounding examples ("how we handled similar cases").
3. A candidate reply that was generated in response to the incoming email.

Score the candidate reply on exactly these 5 dimensions, each on an integer scale from 1 (very poor) to 5 (excellent):
- relevance: does it address what the customer actually asked?
- grounding: does it use only facts present in the incoming email, the retrieved examples, or the stated policies above — with no invented order details, dates, prices, or promises?
- tone: is it empathetic, professional, and consistent with the company's warm support voice?
- actionability: does it give the customer a concrete resolution or next step?
- consistency: is it handled the way the retrieved past replies handled similar cases?

Respond with ONLY valid JSON, no markdown code fences, no prose before or after, matching exactly this schema:
{
  "relevance": {"score": <integer 1-5>, "justification": "<one sentence>"},
  "grounding": {"score": <integer 1-5>, "justification": "<one sentence>"},
  "tone": {"score": <integer 1-5>, "justification": "<one sentence>"},
  "actionability": {"score": <integer 1-5>, "justification": "<one sentence>"},
  "consistency": {"score": <integer 1-5>, "justification": "<one sentence>"}
}"""


def load_json(path):
    with open(path) as f:
        return json.load(f)


def format_email(email):
    return f"From: {email['from']} | Subject: {email['subject']}\n{email['body']}"


def format_retrieved_pairs(pairs):
    blocks = []
    for i, pair in enumerate(pairs, start=1):
        blocks.append(
            f"Retrieved example {i} (category: {pair['category']}):\n"
            f"Customer email — {format_email(pair['incoming_email'])}\n"
            f"Reply that was sent:\n{pair['sent_reply']}"
        )
    return "\n\n".join(blocks)


def build_judge_user_message(incoming_email, retrieved_pairs, candidate_reply):
    return (
        "Incoming customer email:\n"
        f"{format_email(incoming_email)}\n\n"
        "Retrieved past cases used as grounding examples for the candidate reply:\n"
        f"{format_retrieved_pairs(retrieved_pairs)}\n\n"
        "Candidate reply to evaluate:\n"
        f"{candidate_reply}\n\n"
        "Score the candidate reply now, following the rubric and JSON schema from the system prompt."
    )


def extract_json_object(text):
    """Best-effort extraction of a JSON object from a judge response."""
    text = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("no JSON object found in judge response")


def parse_judge_response(text):
    data = extract_json_object(text)
    parsed = {}
    for dim in JUDGE_DIMENSIONS:
        entry = data[dim]
        score = int(round(float(entry["score"])))
        score = max(1, min(5, score))
        justification = str(entry["justification"]).strip()
        parsed[dim] = {"score": score, "justification": justification}
    return parsed


def call_with_retries(fn, max_retries=MAX_RETRIES, sleep_seconds=RETRY_SLEEP_SECONDS):
    """Call fn() with a couple of short-sleep retries on API/rate-limit errors."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except errors.APIError as e:
            if attempt < max_retries:
                print(
                    f"  API error ({e.code}) — retrying in {sleep_seconds}s "
                    f"(attempt {attempt + 1}/{max_retries}) ...",
                    file=sys.stderr,
                )
                time.sleep(sleep_seconds)
            else:
                raise


def call_judge(client, incoming_email, retrieved_pairs, candidate_reply):
    def _call():
        response = client.models.generate_content(
            model=MODEL,
            contents=build_judge_user_message(incoming_email, retrieved_pairs, candidate_reply),
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_SYSTEM_PROMPT,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_TOKENS,
            ),
        )
        return response.text

    text = call_with_retries(_call)
    return parse_judge_response(text)


def run_deterministic_checks(incoming_email, reply):
    checks = {}

    first_line = reply.strip().splitlines()[0] if reply.strip() else ""
    checks["has_greeting"] = bool(GREETING_RE.match(first_line))

    checks["has_signoff"] = bool(SIGNOFF_RE.search(reply.strip()))

    word_count = len(reply.split())
    checks["word_count_in_range"] = WORD_COUNT_MIN <= word_count <= WORD_COUNT_MAX

    checks["no_placeholder_tokens"] = not bool(PLACEHOLDER_RE.search(reply))

    source_text = f"{incoming_email['subject']} {incoming_email['body']}"
    order_match = ORDER_NUMBER_RE.search(source_text)
    if order_match:
        checks["references_order_number"] = order_match.group(0) in reply

    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    pass_rate = passed / total if total else 0.0

    return {
        "checks": checks,
        "word_count": word_count,
        "checks_passed": passed,
        "checks_total": total,
        "pass_rate": round(pass_rate, 4),
    }


def tfidf_cosine_similarity(text_a, text_b):
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform([text_a, text_b])
    except ValueError:
        # Happens if both texts are empty or entirely stop words.
        return 0.0
    return float(cosine_similarity(matrix[0], matrix[1])[0][0])


def compute_composite(judge_scores, deterministic_pass_rate, reference_similarity):
    judge_mean_1to5 = mean(dim["score"] for dim in judge_scores.values())
    judge_pct = (judge_mean_1to5 / 5) * 100
    deterministic_pct = deterministic_pass_rate * 100
    similarity_pct = reference_similarity * 100
    composite = 0.70 * judge_pct + 0.20 * deterministic_pct + 0.10 * similarity_pct
    return round(composite, 2), round(judge_mean_1to5, 3)


def evaluate_response(client, entry, id_to_pair):
    incoming_email = entry["incoming_email"]
    candidate_reply = entry["generated_reply"]

    result = {
        "id": entry["id"],
        "category": entry["category"],
        "incoming_email": incoming_email,
        "generated_reply": candidate_reply,
        "retrieved_example_ids": entry["retrieved_example_ids"],
    }

    if candidate_reply is None:
        result["status"] = "skipped_no_generated_reply"
        return result

    retrieved_pairs = [id_to_pair[rid] for rid in entry["retrieved_example_ids"]]
    reference = id_to_pair[entry["id"]]["sent_reply"]

    deterministic = run_deterministic_checks(incoming_email, candidate_reply)
    result["deterministic_checks"] = deterministic

    reference_similarity = tfidf_cosine_similarity(candidate_reply, reference)
    result["reference_similarity"] = round(reference_similarity, 4)

    try:
        judge_scores = call_judge(client, incoming_email, retrieved_pairs, candidate_reply)
        result["judge"] = judge_scores
    except errors.ClientError as e:
        result["status"] = "judge_error"
        if e.code == 429:
            result["judge_error"] = f"rate limited — {e}"
        elif e.code in (401, 403):
            result["judge_error"] = f"authentication failed — check GEMINI_API_KEY — {e}"
        else:
            result["judge_error"] = f"API client error (status {e.code}) — {e}"
        return result
    except errors.ServerError as e:
        result["status"] = "judge_error"
        result["judge_error"] = f"API server error (status {e.code}) — {e}"
        return result
    except errors.APIError as e:
        result["status"] = "judge_error"
        result["judge_error"] = f"API error (status {e.code}) — {e}"
        return result
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        result["status"] = "judge_parse_error"
        result["judge_error"] = f"could not parse judge response as valid JSON — {e}"
        return result
    except Exception as e:  # noqa: BLE001 - keep the batch running on unexpected errors
        result["status"] = "judge_error"
        result["judge_error"] = f"unexpected error — {e}"
        return result

    composite, judge_mean = compute_composite(
        judge_scores, deterministic["pass_rate"], reference_similarity
    )
    result["judge_mean_score"] = judge_mean
    result["composite_score"] = composite
    result["status"] = "ok"
    return result


def summarize(results):
    scored = [r for r in results if r.get("status") == "ok"]
    if not scored:
        return {
            "responses_evaluated": 0,
            "responses_total": len(results),
            "mean_composite": None,
            "per_category_mean_composite": {},
            "per_dimension_mean_score": {},
        }

    mean_composite = round(mean(r["composite_score"] for r in scored), 2)

    categories = sorted({r["category"] for r in scored})
    per_category_mean_composite = {
        cat: round(mean(r["composite_score"] for r in scored if r["category"] == cat), 2)
        for cat in categories
    }

    per_dimension_mean_score = {
        dim: round(mean(r["judge"][dim]["score"] for r in scored), 3)
        for dim in JUDGE_DIMENSIONS
    }

    return {
        "responses_evaluated": len(scored),
        "responses_total": len(results),
        "mean_composite": mean_composite,
        "per_category_mean_composite": per_category_mean_composite,
        "per_dimension_mean_score": per_dimension_mean_score,
    }


def print_summary_table(results, overall):
    print()
    print("=" * 78)
    print("PER-RESPONSE SUMMARY")
    print("=" * 78)
    header = f"{'id':>4}  {'category':<18}  {'status':<20}  {'composite':>9}  {'judge':>6}  {'det%':>6}  {'sim':>5}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: x["id"]):
        if r.get("status") == "ok":
            det_pct = r["deterministic_checks"]["pass_rate"] * 100
            print(
                f"{r['id']:>4}  {r['category']:<18}  {r['status']:<20}  "
                f"{r['composite_score']:>9.2f}  {r['judge_mean_score']:>6.2f}  "
                f"{det_pct:>5.1f}%  {r['reference_similarity']:>5.3f}"
            )
        else:
            print(f"{r['id']:>4}  {r['category']:<18}  {r.get('status', 'unknown'):<20}  {'--':>9}  {'--':>6}  {'--':>6}  {'--':>5}")

    print()
    print("=" * 78)
    print("OVERALL")
    print("=" * 78)
    print(f"Responses evaluated: {overall['responses_evaluated']}/{overall['responses_total']}")
    print(f"Mean composite score: {overall['mean_composite']}")
    print()
    print("Per-category mean composite:")
    for cat, val in overall["per_category_mean_composite"].items():
        print(f"  {cat:<18} {val}")
    print()
    print("Per-dimension mean judge score (1-5 scale):")
    for dim, val in overall["per_dimension_mean_score"].items():
        print(f"  {dim:<14} {val}")
    print()


def main():
    if not GENERATED_PATH.exists():
        print(
            f"ERROR: {GENERATED_PATH} not found. Run generate_replies.py first "
            "to produce candidate replies before evaluating them.",
            file=sys.stderr,
        )
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset from {DATASET_PATH} ...")
    dataset = load_json(DATASET_PATH)
    id_to_pair = {d["id"]: d for d in dataset}

    print(f"Loading generated replies from {GENERATED_PATH} ...")
    generated = load_json(GENERATED_PATH)
    print(f"Loaded {len(generated)} candidate replies to evaluate.")

    client = genai.Client(api_key=api_key)

    results = []
    for i, entry in enumerate(generated, start=1):
        print(f"[{i}/{len(generated)}] Evaluating response id={entry['id']} category={entry['category']} ...")
        result = evaluate_response(client, entry, id_to_pair)
        status = result.get("status")
        if status == "ok":
            print(f"  composite={result['composite_score']} judge_mean={result['judge_mean_score']}")
        else:
            print(f"  SKIPPED/ERROR: {status} — {result.get('judge_error', 'no generated reply')}")
        results.append(result)

    overall = summarize(results)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({"responses": results, "overall": overall}, f, indent=2)

    print(f"\nSaved full breakdown to {RESULTS_PATH}")
    print_summary_table(results, overall)


if __name__ == "__main__":
    main()
