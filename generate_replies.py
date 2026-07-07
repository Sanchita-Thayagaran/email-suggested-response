#!/usr/bin/env python3
"""Generate suggested replies for test-split support emails using retrieval + Gemini."""

import json
import os
import sys
import time
from pathlib import Path

from google import genai
from google.genai import errors, types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.json"
OUTPUT_PATH = BASE_DIR / "outputs" / "generated.json"

MODEL = "gemini-2.5-flash"
MAX_TOKENS = 1024
TOP_K = 3

MAX_RETRIES = 4
RETRY_SLEEP_SECONDS = 2

SYSTEM_PROMPT = """You are a customer support agent for Northwind Gear, an outdoor-gear e-commerce company (support@northwindgear.com).

Company policies (these are the only policies that exist — do not invent others):
- Returns: 30-day return window from the delivery date, full refund to the original payment method.
- Warranty: 2-year warranty on backpacks ("packs") against manufacturing defects. No other product category carries a stated warranty.
- Shipping: standard shipping takes 5-7 business days.

Voice and structure for every reply:
- Warm, professional tone.
- Structure: greeting -> resolution -> concrete next step -> sign-off.
- Sign off with exactly: "— Northwind Gear Support"
- Reference specifics from the incoming email (customer name, order number, product name) when they are given.

Grounding rules (critical):
- Base the reply only on facts stated in the incoming email and in the example replies provided to you.
- Never invent order numbers, tracking numbers, dates, prices, or any other specifics that are not given to you.
- If information needed to fully resolve the request is missing, ask for it rather than guessing.
- Match the tone, structure, and level of detail demonstrated in the example replies."""


def load_dataset(path):
    with open(path) as f:
        return json.load(f)


def email_text(incoming_email):
    return f"{incoming_email['subject']}\n\n{incoming_email['body']}"


def build_retrieval_index(train_pairs):
    corpus = [email_text(pair["incoming_email"]) for pair in train_pairs]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def retrieve_top_k(vectorizer, matrix, train_pairs, query_email, k=TOP_K):
    query_vector = vectorizer.transform([email_text(query_email)])
    scores = cosine_similarity(query_vector, matrix)[0]
    ranked = scores.argsort()[::-1][:k]
    return [(train_pairs[i], float(scores[i])) for i in ranked]


def format_examples_block(examples):
    blocks = []
    for i, (pair, _score) in enumerate(examples, start=1):
        incoming = pair["incoming_email"]
        blocks.append(
            f"Example {i} — how we handled a similar case:\n"
            f"Customer email — From: {incoming['from']} | Subject: {incoming['subject']}\n"
            f"{incoming['body']}\n\n"
            f"Reply we sent:\n{pair['sent_reply']}"
        )
    return "\n\n".join(blocks)


def build_user_message(test_pair, examples):
    incoming = test_pair["incoming_email"]
    return (
        "How we handled similar cases (use these for grounding and to match voice, "
        "but do not copy over any facts specific to them):\n\n"
        f"{format_examples_block(examples)}\n\n"
        "New incoming email to respond to:\n"
        f"From: {incoming['from']} | Subject: {incoming['subject']}\n"
        f"{incoming['body']}\n\n"
        "Write the reply Northwind Gear Support should send to this new email. "
        "Follow the grounding rules and voice from the system prompt."
    )


def extract_retry_delay_seconds(e):
    """Pull the server-suggested retry delay (e.g. "24s") out of a 429 error.

    The free tier's RESOURCE_EXHAUSTED response includes a RetryInfo detail
    telling us exactly how long to wait — a fixed short sleep isn't enough
    for a 5-requests-per-minute quota, so we honor the real number when Gemini
    gives us one and fall back to a fixed sleep only if it's missing/unparseable.
    """
    try:
        for detail in e.details.get("error", {}).get("details", []):
            if detail.get("@type", "").endswith("RetryInfo"):
                delay = detail.get("retryDelay", "")
                if delay.endswith("s"):
                    return float(delay[:-1])
    except (AttributeError, TypeError, ValueError):
        pass
    return None


def call_with_retries(fn, max_retries=MAX_RETRIES, sleep_seconds=RETRY_SLEEP_SECONDS):
    """Call fn() with a few retries on API/rate-limit errors.

    On 429s, sleeps for the server-reported retryDelay (plus a small buffer)
    rather than a fixed short sleep, since the free tier's quota window can
    take much longer than a couple of seconds to reset.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except errors.APIError as e:
            if attempt < max_retries:
                wait = sleep_seconds
                if e.code == 429:
                    retry_delay = extract_retry_delay_seconds(e)
                    if retry_delay is not None:
                        wait = retry_delay + 1
                print(
                    f"  API error ({e.code}) — retrying in {wait:.0f}s "
                    f"(attempt {attempt + 1}/{max_retries}) ...",
                    file=sys.stderr,
                )
                time.sleep(wait)
            else:
                raise


def generate_reply(client, test_pair, examples):
    def _call():
        response = client.models.generate_content(
            model=MODEL,
            contents=build_user_message(test_pair, examples),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=MAX_TOKENS,
                # Disabled defensively — thinking tokens share max_output_tokens
                # with the final answer on Gemini 2.5 Flash, which was observed
                # truncating judge JSON in evaluate.py under the same setup.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text.strip()

    return call_with_retries(_call)


def process_test_email(client, test_pair, vectorizer, matrix, train_pairs):
    examples = retrieve_top_k(vectorizer, matrix, train_pairs, test_pair["incoming_email"])
    retrieved_ids = [pair["id"] for pair, _score in examples]
    retrieval_scores = [round(score, 4) for _pair, score in examples]
    print(f"  Retrieved train ids: {retrieved_ids} (scores: {retrieval_scores})")

    generated_reply = None
    try:
        generated_reply = generate_reply(client, test_pair, examples)
        print("  Generated reply successfully.")
    except errors.ClientError as e:
        if e.code == 429:
            print(f"  ERROR: rate limited — {e}", file=sys.stderr)
        elif e.code in (401, 403):
            print(f"  ERROR: authentication failed — check GEMINI_API_KEY — {e}", file=sys.stderr)
        else:
            print(f"  ERROR: API client error (status {e.code}) — {e}", file=sys.stderr)
    except errors.ServerError as e:
        print(f"  ERROR: API server error (status {e.code}) — {e}", file=sys.stderr)
    except errors.APIError as e:
        print(f"  ERROR: API error (status {e.code}) — {e}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 - keep the batch running on unexpected errors
        print(f"  ERROR: unexpected error — {e}", file=sys.stderr)

    return {
        "id": test_pair["id"],
        "category": test_pair["category"],
        "incoming_email": test_pair["incoming_email"],
        "retrieved_example_ids": retrieved_ids,
        "retrieval_scores": retrieval_scores,
        "generated_reply": generated_reply,
    }


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset from {DATASET_PATH} ...")
    data = load_dataset(DATASET_PATH)
    train_pairs = [d for d in data if d["split"] == "train"]
    test_pairs = [d for d in data if d["split"] == "test"]
    print(f"Loaded {len(train_pairs)} train pairs (retrieval corpus) and {len(test_pairs)} test emails to answer.")

    print("Building TF-IDF retrieval index over train incoming emails (subject + body) ...")
    vectorizer, matrix = build_retrieval_index(train_pairs)

    client = genai.Client(api_key=api_key)

    results = []
    for i, test_pair in enumerate(test_pairs, start=1):
        print(f"[{i}/{len(test_pairs)}] Processing test email id={test_pair['id']} category={test_pair['category']} ...")
        results.append(process_test_email(client, test_pair, vectorizer, matrix, train_pairs))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    succeeded = sum(1 for r in results if r["generated_reply"] is not None)
    print(f"Done. {succeeded}/{len(results)} replies generated successfully.")
    print(f"Saved output to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
