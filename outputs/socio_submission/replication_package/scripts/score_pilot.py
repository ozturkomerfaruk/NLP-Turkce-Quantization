"""Evaluate a 20-pair pilot for one EuroLLM precision and explain its scoring path.

Inspect the first pair at token level, verify the shared scoring function, then
score all pilot pairs and save their results and summary. Positive mean-logp
margins count as correct; exact ties do not. Turkish text remains benchmark data."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

# Evaluate one precision per process, using the same pilot pairs for every model.
parser = argparse.ArgumentParser(description="TurBLiMP 20-pair pilot evaluation")
parser.add_argument("--model", required=True, choices=["fp16", "q8", "q4"])
args = parser.parse_args()
MODEL_KEY = args.model
PRECISION = MODEL_KEY.upper()

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / f"eurollm-{MODEL_KEY}"
PILOT_PATH = ROOT / "data" / "processed" / "turblimp_pilot_20.jsonl"

EXPECTED_SHA256 = (
    "420407b129d0e997680a50ab11ff49319e933c1b82b07068042b5d1249239f27"
)

print(f"{PRECISION} model exists:", (MODEL_PATH / "config.json").is_file())
print("Pilot dataset exists:", PILOT_PATH.is_file())

if not (MODEL_PATH / "config.json").is_file():
    raise FileNotFoundError(f"Model configuration not found: {MODEL_PATH / 'config.json'}")
if not PILOT_PATH.is_file():
    raise FileNotFoundError(f"Pilot dataset not found: {PILOT_PATH}")


# Use the first nonempty record as a worked example of the scoring protocol.
def load_first_pair():
    with PILOT_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                return json.loads(line)

    raise ValueError("The pilot dataset file is empty.")


# Score every sentence token with BOS context, without adding or scoring EOS.
def score_sentence(model, tokenizer, sentence):
    token_ids = tokenizer.encode(
        sentence,
        add_special_tokens=False
    )

    if not token_ids:
        raise ValueError("The sentence is empty or could not be tokenized.")

    bos_id = tokenizer.bos_token_id

    if bos_id is None:
        raise ValueError("BOS token not found.")

    sequence = [bos_id] + token_ids

    inputs = mx.array(
        [sequence[:-1]],
        dtype=mx.int32
    )

    targets = mx.array(
        [sequence[1:]],
        dtype=mx.int32
    )

    logits = model(inputs)

    log_probs = nn.log_softmax(
        logits.astype(mx.float32),
        axis=-1
    )

    selected = mx.take_along_axis(
        log_probs,
        targets[..., None],
        axis=-1
    )

    total_score = mx.sum(selected)
    average_score = total_score / len(token_ids)

    mx.eval(total_score, average_score)

    return {
        "token_count": len(token_ids),
        "total": total_score.item(),
        "average": average_score.item()
    }


# Check the frozen pilot hash before interpreting any scores as comparable.
print("\n--- DATASET INTEGRITY CHECK ---")

dataset_sha256 = hashlib.sha256(
    PILOT_PATH.read_bytes()
).hexdigest()

print("Expected SHA-256:", EXPECTED_SHA256)
print("Computed SHA-256:", dataset_sha256)

if dataset_sha256 != EXPECTED_SHA256:
    raise ValueError("Pilot dataset SHA-256 mismatch!")

print("SHA-256 verification passed.")


pair = load_first_pair()

print("\nFirst sentence pair:")
print(pair)
print("Data type:", type(pair))

good = pair["sentence_good"]
bad = pair["sentence_bad"]
pair_id = pair["pair_id"]
category = pair["category"]
critical_region = pair["critical_region"]

print("\n--- FIRST EXPERIMENT PAIR ---")
print("Pair ID:", pair_id)
print("Category:", category)
print("Critical region:", critical_region)

print("\nGood sentence:", good)
print("Bad sentence:", bad)

print(f"\n{PRECISION} model loading...")

model, tokenizer = load(str(MODEL_PATH))

model.eval()  # Use inference mode consistently across model precisions.

print(f"{PRECISION} model loaded successfully.")

# Inspect one worked example to expose subword boundaries before scoring the pilot.
token_ids = tokenizer.encode(
    good,
    add_special_tokens=False
)

print("\nGood sentence:", good)
print("Token IDs:", token_ids)
print("Token count:", len(token_ids))

tokens = tokenizer.convert_ids_to_tokens(token_ids)

print("\n--- TOKEN ANALYSIS ---")

for index, (token_id, token) in enumerate(
    zip(token_ids, tokens), start=1
):
    print(index, token_id, repr(token))

bos_id = tokenizer.bos_token_id

if bos_id is None:
    raise ValueError("Tokenizer has no BOS token.")

print("\nBOS Token:", tokenizer.bos_token)
print("BOS ID:", bos_id)

# Shift inputs and targets by one position: BOS predicts the first sentence token.
sequence = [bos_id] + token_ids

print("\nSequence:", sequence)
print("Sequence length:", len(sequence))

inputs = sequence[:-1]
targets = sequence[1:]

print("\n--- INPUT / TARGET ALIGNMENT ---")

for index, (input_id, target_id) in enumerate(
    zip(inputs, targets), start=1
):
    input_token = tokenizer.convert_ids_to_tokens(input_id)
    target_token = tokenizer.convert_ids_to_tokens(target_id)

    print(
        f"{index}. {input_token} -> {target_token}"
    )


input_tensor = mx.array(
    [inputs],
    dtype=mx.int32
)

target_tensor = mx.array(
    [targets],
    dtype=mx.int32
)

print("\n--- TENSOR CHECK ---")

print("Input Shape:", input_tensor.shape)
print("Target Shape:", target_tensor.shape)
print("Input Dtype:", input_tensor.dtype)

print(f"\n{PRECISION} forward pass starting...")
logits = model(input_tensor)
mx.eval(logits)

print("\n--- FORWARD PASS RESULT ---")

print("Logits Shape:", logits.shape)
print("Logits Dtype:", logits.dtype)

print("Batch Size:", logits.shape[0])
print("Sequence Length:", logits.shape[1])
print("Vocabulary Size:", logits.shape[2])

# Convert logits to float32 log-probabilities and gather only observed target tokens.
log_probs = nn.log_softmax(
    logits.astype(mx.float32),
    axis=-1
)

print("\nLog Probabilities Shape:", log_probs.shape)
print("Log Probabilities Dtype:", log_probs.dtype)

selected = mx.take_along_axis(
    log_probs,
    target_tensor[..., None],
    axis=-1
)

token_log_probs = mx.squeeze(selected, axis=-1)

total_score = mx.sum(token_log_probs)
average_score = total_score / len(targets)
mx.eval(total_score, average_score)

print(f"\n--- {PRECISION} TOKEN SCORES ---")

scores = token_log_probs[0].tolist()

for token, score in zip(tokens, scores):
    print(f"{repr(token):15} {score:.4f}")

print(f"\n--- {PRECISION} SENTENCE SCORE ---")

print("Token count:", len(targets))
print("Total log-probability:", total_score.item())
print("Mean log-probability:", average_score.item())


# Score the bad sentence independently so different token lengths remain valid.
bad_token_ids = tokenizer.encode(
    bad,
    add_special_tokens=False
)

print("\n--- BAD SENTENCE ---")
print("Sentence:", bad)
print("Token IDs:", bad_token_ids)
print("Token count:", len(bad_token_ids))

bad_sequence = [bos_id] + bad_token_ids

bad_inputs = mx.array(
    [bad_sequence[:-1]],
    dtype=mx.int32
)

bad_targets = mx.array(
    [bad_sequence[1:]],
    dtype=mx.int32
)

bad_logits = model(bad_inputs)

bad_log_probs = nn.log_softmax(
    bad_logits.astype(mx.float32),
    axis=-1
)

bad_selected = mx.take_along_axis(
    bad_log_probs,
    bad_targets[..., None],
    axis=-1
)

bad_token_scores = mx.squeeze(
    bad_selected,
    axis=-1
)

bad_total_score = mx.sum(bad_token_scores)

bad_average_score = (
    bad_total_score / len(bad_token_ids)
)

mx.eval(bad_total_score, bad_average_score)


good_score = average_score.item()
bad_score = bad_average_score.item()

difference = good_score - bad_score

print(f"\n--- {PRECISION} COMPARISON ---")

print(f"Good sentence: {good_score:.6f}")
print(f"Bad sentence: {bad_score:.6f}")
print(f"Score difference: {difference:.6f}")

if difference > 0:
    print("Decision: CORRECT")
elif difference < 0:
    print("Decision: INCORRECT")
else:
    print("Decision: TIE")


bad_tokens = tokenizer.convert_ids_to_tokens(
    bad_token_ids
)

good_scores = token_log_probs[0].tolist()
bad_scores = bad_token_scores[0].tolist()

print("\n--- TOKEN-LEVEL COMPARISON ---")

# Compare positions only when token counts match. Even equal counts do not
# establish linguistic alignment: subword boundaries may differ between sentences.
if len(tokens) == len(bad_tokens):
    print("Note: Only tokens at the same positions are compared;")
    print("different token boundaries do not imply linguistic alignment.")
    print(
        f"{'No':<4}"
        f"{'Good token':<15}"
        f"{'Bad token':<15}"
        f"{'Good':>10}"
        f"{'Bad':>10}"
        f"{'Difference':>10}"
    )

    for i, (good_token, bad_token, good_log, bad_log) in enumerate(
        zip(tokens, bad_tokens, good_scores, bad_scores),
        start=1
    ):
        token_difference = good_log - bad_log

        print(
            f"{i:<4}"
            f"{good_token:<15}"
            f"{bad_token:<15}"
            f"{good_log:>10.4f}"
            f"{bad_log:>10.4f}"
            f"{token_difference:>+10.4f}"
        )
else:
    print("Token counts differ: positional comparison skipped.")

print("\n--- INDEPENDENT TOKEN-LEVEL ANALYSIS ---")

print("\nGOOD SENTENCE TOKENS")
print(f"{'No':<5}{'Token':<20}{'Log P':>12}")

for i, (token, score) in enumerate(
    zip(tokens, good_scores), start=1
):
    print(f"{i:<5}{token:<20}{score:>12.4f}")

print("\nBAD SENTENCE TOKENS")
print(f"{'No':<5}{'Token':<20}{'Log P':>12}")

for i, (token, score) in enumerate(
    zip(bad_tokens, bad_scores), start=1
):
    print(f"{i:<5}{token:<20}{score:>12.4f}")

print("\n--- TOKEN COUNT CHECK ---")
print("Good sentence token count:", len(tokens))
print("Bad sentence token count:", len(bad_tokens))

if len(tokens) != len(bad_tokens):
    print(
        "Token counts differ. "
        "Positional token comparison is skipped."
    )


# Cross-check the reusable scorer against the explicit first-pair calculation.

print("\n--- FUNCTION TEST ---")

good_result = score_sentence(model, tokenizer, good)
bad_result = score_sentence(model, tokenizer, bad)

print("Good sentence:", good_result)
print("Bad sentence:", bad_result)

assert abs(good_result["average"] - good_score) < 1e-4
assert abs(bad_result["average"] - bad_score) < 1e-4

print("Function verification passed.")

print(f"\n--- {PRECISION}: 20-PAIR PILOT EVALUATION ---")

with PILOT_PATH.open("r", encoding="utf-8") as file:
    pilot_pairs = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

if len(pilot_pairs) != 20:
    raise ValueError(
        f"Expected 20 pairs; {len(pilot_pairs)} found."
    )

# Apply the verified scorer to every pilot pair; mean logp is the primary decision score.
pilot_results = []

for index, current_pair in enumerate(pilot_pairs, start=1):

    good_result = score_sentence(
        model,
        tokenizer,
        current_pair["sentence_good"]
    )

    bad_result = score_sentence(
        model,
        tokenizer,
        current_pair["sentence_bad"]
    )

    delta = good_result["average"] - bad_result["average"]

    correct = delta > 0

    result = {
        "pair_id": current_pair["pair_id"],
        "category": current_pair["category"],
        "model": PRECISION,
        "dataset_sha256": dataset_sha256,
        "scoring_method": "mean_token_log_probability",
        "good_score": good_result["average"],
        "good_total_score": good_result["total"],
        "bad_score": bad_result["average"],
        "bad_total_score": bad_result["total"],
        "good_token_count": good_result["token_count"],
        "bad_token_count": bad_result["token_count"],
        "delta": delta,
        "correct": correct
    }

    pilot_results.append(result)

    print(
        f"{index:02d}/20 | "
        f"{current_pair['pair_id']} | "
        f"Difference: {delta:+.6f} | "
        f"{'CORRECT' if correct else 'INCORRECT'}"
    )

# Summarize exact ties separately while retaining them among unsuccessful pairs.
success_count = sum(
    result["correct"]
    for result in pilot_results
)

tie_count = sum(
    result["delta"] == 0
    for result in pilot_results
)

accuracy = success_count / len(pilot_results)

print(f"\n--- {PRECISION} PILOT SUMMARY ---")
print("Total pairs:", len(pilot_results))
print("Correct pairs:", success_count)
print("Incorrect pairs:", len(pilot_results) - success_count)
print("Ties:", tie_count)
print(f"Accuracy: %{accuracy * 100:.2f}")

category_results = {}
for result in pilot_results:
    key = result["category"]
    if key not in category_results:
        category_results[key] = {"total": 0, "correct": 0}
    category_results[key]["total"] += 1
    category_results[key]["correct"] += int(result["correct"])

print("\n--- RESULTS BY CATEGORY ---")
for category_name, values in category_results.items():
    category_accuracy = 100 * values["correct"] / values["total"]
    print(f"{category_name}: {values['correct']}/{values['total']} (%{category_accuracy:.2f})")

RESULTS_DIR = ROOT / "results" / "pilot"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_PATH = RESULTS_DIR / f"{MODEL_KEY}_pilot_20.jsonl"

with RESULTS_PATH.open("w", encoding="utf-8") as file:
    for result in pilot_results:
        file.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )

print("\n--- RESULTS SAVED ---")
print("File:", RESULTS_PATH)
print("Saved record count:", len(pilot_results))

SUMMARY_PATH = RESULTS_DIR / f"{MODEL_KEY}_summary.json"

# Save provenance and tokenization rules alongside accuracy to make comparisons auditable.
summary = {
    "model": "EuroLLM-1.7B Base",
    "precision": PRECISION,
    "model_path": str(MODEL_PATH),
    "dataset_path": str(PILOT_PATH),
    "dataset_sha256": dataset_sha256,
    "scoring_method": "mean_token_log_probability",
    "scoring_dtype": "float32",
    "tokenization": {
        "add_special_tokens": False,
        "prepend_bos": True,
        "score_bos": False,
        "append_eos": False
    },
    "environment": {
        "python_version": platform.python_version(),
        "mlx_version": version("mlx"),
        "mlx_lm_version": version("mlx-lm"),
        "platform": platform.platform()
    },
    "total_pairs": len(pilot_results),
    "successful_pairs": success_count,
    "unsuccessful_pairs": len(pilot_results) - success_count,
    "tie_count": tie_count,
    "accuracy": accuracy,
    "category_results": category_results
}

with SUMMARY_PATH.open("w", encoding="utf-8") as file:
    json.dump(
        summary,
        file,
        ensure_ascii=False,
        indent=4
    )

print("Summary file:", SUMMARY_PATH)
