# Reproduce the revision experiment

The package includes 3,662 TurBLiMP minimal pairs and 444 verified checkpoints from six model configurations. The data manifest distinguishes the 1,000 previously evaluated pairs from the remaining 2,662. The [analysis protocol](revision_analysis_protocol_v1.md) and [statistical methods](../results/revision_v2/statistics/methods.md) define the comparisons.

Start from the root of a clone checked out at the commit cited for the manuscript or from an extracted review archive. If using Git, record the output of `git rev-parse HEAD` when reporting a reproduction. The author's reference scores and tables are in [`results/revision_v2/`](../results/revision_v2/). Generated weights, run logs and local reproduction files are ignored by Git.

## Manuscript table sources

| Table | Reading the reference outputs |
| --- | --- |
| 1. Pair counts | [Dataset manifest](../data/revision_v2/manifest.json) and its category, duplicate and sample membership counts |
| 2. Accuracy and paired Q4 effects | [Micro and macro accuracy](../results/revision_v2/statistics/accuracy.csv) and [paired effects and transitions](../results/revision_v2/statistics/paired.csv), using `full_3662` |
| 3. Q8 equivalence | [RQ5 template-clustered tests](../results/revision_v2/statistics/tests.csv) and preference counts from [paired effects](../results/revision_v2/statistics/paired.csv), using `full_3662` |
| 4. Q4 category heterogeneity | [RQ3 template-clustered `category_omnibus` tests](../results/revision_v2/statistics/tests.csv) for `full_3662` and `new_2662` |
| 5. Q4 effects by category | [Q4 category effects](../results/revision_v2/statistics/q4_category_effects.csv), using `full_3662` |
| 6. Resource measurements | [Resource measurements](../results/revision_v2/diagnostics/resource_measurements.csv). Divide byte values by 10⁹ for GB. |

The scoring and length analysis uses [the length table](../results/revision_v2/statistics/length.csv). The exploratory predicate check uses [the flagged pairs](../results/revision_v2/diagnostics/predicate_screen_pairs.csv) and [the sensitivity table](../results/revision_v2/diagnostics/predicate_sensitivity.csv).

| Goal | Requirements |
| --- | --- |
| Check the dataset and score archive | Python 3.12 on macOS or Linux, including WSL on Windows |
| Recompute statistics from saved scores | Python 3.12, R 4.5.3 and clubSandwich 0.7.0 |
| Rebuild the six models and repeat scoring | Apple Silicon macOS with Metal, Python 3.12.15, `requirements.txt`, internet access and at least 25 GB of free disk space |

The reference experiment used an Apple M2 with 16 GB of unified memory. Other inference backends can change the reconstructed model artifacts and numerical output. This package documents the original MLX path.

## Verify the saved scores

```bash
python3 -m unittest discover -s tests -v
mkdir -p outputs/revision_v2/reproduction
python3 scripts/prepare_revision_analysis.py \
  --output outputs/revision_v2/reproduction/pairs.csv
python3 scripts/compare_revision_scores.py results/revision_v2 results/revision_v2
```

The exporter validates the score checkpoints against [the audit](../results/revision_v2/verification.json) and produces 3,662 aligned rows and 2,273 approximate template clusters. The score archive compared with itself should give 12 rows with zero preference changes and zero score differences.

## Recompute tables from saved scores

On Apple Silicon, recreate the R environment with micromamba. Install the verified CRAN source package into this environment.

```bash
micromamba create -y -n turblimp-revision-r --file docs/revision_r_osx_arm64.explicit
mkdir -p outputs/revision_v2/reproduction
package=outputs/revision_v2/reproduction/clubSandwich_0.7.0.tar.gz
curl --fail --location --output "$package" \
  https://cran.r-project.org/src/contrib/clubSandwich_0.7.0.tar.gz ||
  curl --fail --location --output "$package" \
    https://cran.r-project.org/src/contrib/Archive/clubSandwich/clubSandwich_0.7.0.tar.gz
printf '%s  %s\n' f3cd9cd5840022b8d1354dce3e7e4a820edcd47411b9f95bdd200c4e040032ed "$package" |
  shasum -a 256 --check && micromamba run -n turblimp-revision-r R CMD INSTALL "$package"
```

On other platforms, install the versions in the [R session record](../results/revision_v2/statistics/session-info.txt). With that environment on your `PATH`, use `Rscript` in place of `micromamba run -n turblimp-revision-r Rscript` below.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 micromamba run -n turblimp-revision-r Rscript \
  scripts/analyze_revision.R outputs/revision_v2/reproduction/pairs.csv \
  outputs/revision_v2/reproduction/tables
python3 scripts/verify_revision_tables.py outputs/revision_v2/reproduction/tables
python3 scripts/summarize_revision_diagnostics.py
```

The comparator checks nine CSVs against the archived tables, with a tolerance for floating-point arithmetic. The two scoring rules, model families, and full and smaller evaluation subsets are shown separately. The diagnostic script regenerates the three CSVs in `results/revision_v2/diagnostics/` from the archived metrics. These timing and memory values describe the archived execution.

## Rebuild the data and models

```bash
python3 scripts/prepare_revision_dataset.py --output-dir outputs/revision_v2/rebuilt-data
```

The builder validates the four source CSVs and the balanced subset. Expect 3,662 pairs and dataset SHA-256 `f04ee6b48fa27056261e85741139f64a59e1c4078928f2855fdf381cc30c94e9`. The [data manifest](../data/revision_v2/manifest.json) records the exact inputs and category counts.

| Model | Pinned upstream commit | Commands and hashes |
| --- | --- | --- |
| EuroLLM-1.7B | `e00c97a9a389cc184174163cff4ba58facdb8e28` | [Build](eurollm_revision_build.md), [conversion trace](eurollm_revision_conversion_trace.txt), [manifest](eurollm_revision_manifest.json) |
| Qwen3-1.7B-Base | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` | [Build](qwen_revision_build.md), [conversion trace](qwen_revision_conversion_trace.txt), [manifest](qwen_revision_manifest.json) |

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
bash scripts/rebuild_eurollm_revision.sh
bash scripts/build_qwen_revision.sh
```

Each build validates the upstream weight hash, produces FP16 from the pinned source, and converts Q8 and Q4 separately from that FP16 reference using affine quantization with group size 64. The tokenizer, configuration and weight indexes are already present in [the model directories](../models/revision_v2/). The build compares each converted metadata file with the checked-out file before adding the locally generated weights. The scripts update the two local model manifests after verification. Git retains the reference manifests in the checked-out commit.

## Repeat scoring and compare with the archived experiment

Use a separate clone or an extracted review archive for fresh scoring. Move the complete reference results directory, including its audit and tables, to a local comparison location before the first scoring command. The guard preserves the reference copy when resuming later.

```bash
mkdir -p outputs/revision_v2/reproduction
if [ ! -e outputs/revision_v2/reproduction/reference ]; then
  mv results/revision_v2 outputs/revision_v2/reproduction/reference
fi
```

Run the following from the root of that copy. The scorer checks the required eight-pair smoke run before each full configuration. Completed checkpoints with matching provenance are reused after interruption.

```bash
for model in eurollm qwen; do
  for precision in fp16 q8 q4; do
    .venv/bin/python scripts/score_revision.py \
      --model "$model" --precision "$precision" --mode smoke --max-chunks 2 || exit 1
    .venv/bin/python scripts/score_revision.py \
      --model "$model" --precision "$precision" --mode main --max-chunks 74 || exit 1
  done
done
.venv/bin/python scripts/verify_revision_results.py
python3 scripts/compare_revision_scores.py \
  outputs/revision_v2/reproduction/reference results/revision_v2 \
  > outputs/revision_v2/reproduction/score-comparison.csv
```

The comparison validates checkpoint hashes against both audits and aligns pairs and token sequences. Each of the 12 CSV rows reports the author's and reviewer's correct counts, any changes in sentence preference, and the largest absolute sentence score difference. Inspect `preference_changes` and `max_abs_sentence_score_delta` even when the correct counts match. Weight hashes, dataset hash and scoring-code hashes must match. Timing and memory are measured separately on each run.

To compare the reproduced statistics, regenerate `pairs.csv` and run the R command above with a separate output directory, such as `outputs/revision_v2/reproduction/rescored-tables`. Then compare it with the original archive.

```bash
python3 scripts/verify_revision_tables.py outputs/revision_v2/reproduction/rescored-tables \
  --reference outputs/revision_v2/reproduction/reference/statistics
```
