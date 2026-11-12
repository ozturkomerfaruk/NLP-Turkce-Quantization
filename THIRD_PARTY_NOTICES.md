# Third-party materials

The root MIT license covers only original project code and original accompanying technical documentation. It does not replace the following upstream terms or claim ownership of third-party materials.

## TurBLiMP data — CC BY 4.0

- Authors: Ezgi Başar, Federico Padovani, Jaap Jumelet, Arianna Bisazza (2025).
- Paper: https://doi.org/10.18653/v1/2025.emnlp-main.834
- Source: https://github.com/ezgibasar/TurBLiMP
- Data revision: `297de13fb7a0ce524fe32e8b175c6b5255d66960`.
- License: https://creativecommons.org/licenses/by/4.0/
- Applies to the source CSVs in `data/raw/` and derived sentence samples in `data/processed/`.
- Changes: exact duplicate removal within categories and balanced sampling (seed 42); see `data/processed/manifest.json` for provenance and hashes.

## EuroLLM model assets — Apache-2.0

- Source: https://huggingface.co/utter-project/EuroLLM-1.7B
- Paper: https://doi.org/10.48550/arXiv.2409.16235
- License text: [Apache-2.0](licenses/Apache-2.0.txt) (official source: https://www.apache.org/licenses/LICENSE-2.0).
- Applies to upstream model cards, configuration and tokenizer assets in `models/`, including copies used for the FP16/Q8/Q4 configurations. Original model cards and notices are retained.
- Local Q8/Q4 configurations reflect MLX affine quantization with group size 64. No model weights are distributed here.
- The download reference recorded by the project is `refs/pr/5`; an immutable upstream model revision and complete conversion commands were not archived. See the README for the resulting reproducibility limitation.

Python dependencies retain their own licenses. The project MIT license does not grant rights to dependency implementations or to the manuscript's publication content.
