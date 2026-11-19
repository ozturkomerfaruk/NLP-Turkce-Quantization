# Third-party materials

The root MIT license covers only original project code and original accompanying technical documentation. It does not replace the following upstream terms or claim ownership of third-party materials.

## TurBLiMP data — CC BY 4.0

- Authors: Ezgi Başar, Federico Padovani, Jaap Jumelet, Arianna Bisazza (2025).
- Paper: https://doi.org/10.18653/v1/2025.emnlp-main.834
- Source: https://github.com/ezgibasar/TurBLiMP
- Data revision: `297de13fb7a0ce524fe32e8b175c6b5255d66960`.
- License: https://creativecommons.org/licenses/by/4.0/
- Applies to the source CSVs in `data/raw/` and derived sentence samples in `data/processed/` and `data/revision_v2/`.
- Changes: exact duplicate removal within categories, balanced sampling for the original experiment (seed 42), and inclusion of all 3,662 eligible pairs with original-sample membership in the revision. See the manifest in each processed data directory.

## EuroLLM model assets — Apache-2.0

- Source: https://huggingface.co/utter-project/EuroLLM-1.7B
- Paper: https://doi.org/10.48550/arXiv.2409.16235
- License text: [Apache-2.0](licenses/Apache-2.0.txt) (official source: https://www.apache.org/licenses/LICENSE-2.0).
- Applies to upstream model cards, configuration and tokenizer assets in `models/`, including copies used for the FP16/Q8/Q4 configurations. Original model cards and notices are retained.
- Local Q8/Q4 configurations reflect MLX affine quantization with group size 64. No model weights are distributed here.
- The original experiment recorded `refs/pr/5`. The revision uses upstream commit `e00c97a9a389cc184174163cff4ba58facdb8e28`. See [conversion and verification records](docs/eurollm_revision_build.md).

## Qwen3-1.7B-Base — Apache-2.0

- Source: https://huggingface.co/Qwen/Qwen3-1.7B-Base/tree/ea980cb0a6c2ae4b936e82123acc929f1cec04c1
- The source tokenizer and configurations are copied into [the revision model directory](models/revision_v2/qwen/). Its FP16, Q8 and Q4 metadata was generated with MLX, using affine quantization with group size 64 for Q8 and Q4. See the [conversion and verification records](docs/qwen_revision_build.md) and the [Apache-2.0 text](licenses/Apache-2.0.txt).

Python dependencies retain their own licenses. The project MIT license does not grant rights to dependency implementations or to the manuscript's publication content.
