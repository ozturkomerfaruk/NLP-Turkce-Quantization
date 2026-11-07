# EuroLLM Turkish grammatical judgments replication materials

Package preparation date: 7 November 2026. The identifying public repository link and inspected revision are supplied separately on the manuscript title page for anonymous review. The supplementary package additionally contains the selected sample and exploratory analyses; it has no DOI.

## Scope
The package contains the original selected TurBLiMP sample, source CSVs, saved pair-level model scores, statistical outputs, analysis code, and model configuration files. It contains no model weights. It supports reanalysis of the reported benchmark experiment; the evaluation consists of the completed minimal-pair experiment and analyses of its saved scores.

## Provenance and attribution
TurBLiMP: Başar, Padovani, Jumelet, and Bisazza (2025), https://doi.org/10.18653/v1/2025.emnlp-main.834 . Repository: https://github.com/ezgibasar/TurBLiMP . Selected source revision: 297de13fb7a0ce524fe32e8b175c6b5255d66960. The repository README states CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The source CSVs are retained; the JSONL sample is a deduplicated, sampled, reformatted derivative. Credit the original authors and indicate those modifications when redistributing.

EuroLLM: Martins et al. (2024), https://doi.org/10.48550/arXiv.2409.16235 . Model card: https://huggingface.co/utter-project/EuroLLM-1.7B . Model weights are excluded. MLX documentation: https://ml-explore.github.io/mlx/ . Do not infer a complete conversion recipe from the configuration files; upstream model revision and conversion commands were incompletely archived.

The author's own code has no newly assigned license in this preparation. Decide the code license before public deposit. Existing upstream rights remain unchanged.

## Reanalysis without model inference
From the extracted package root, run:

    python3 scripts/analyze_main.py

This reads frozen scores and regenerates results/analysis outputs. It does not rerun the model. Without the weight files, this command reports null for the weight-file size fields; the other regenerated main-statistics values were checked against the original and matched. The archive retains the original measured size fields in the saved outputs. Preserve an untouched copy of the archive if comparing regenerated and saved outputs. Main analysis uses Python's standard library. The original inference used Python 3.12.15 and the additional analyses report NumPy 2.5.4, SciPy 1.18.1, and Matplotlib 3.11.2; environment metadata are under results/deep_analysis.

To recompute exploratory fragmentation estimates, use a compatible scientific-Python environment and run:

    python3 scripts/analyze_fragmentation.py --bootstrap 2000

The extended forensic audit script requires the original weight files for tensor inspection and size calculations. It is retained for transparency, but cannot be fully rerun from this weights-excluded package. The historical provenance inventory may name files outside the package; PACKAGE_SHA256.json is the inventory of files actually supplied here.

## Use with the manuscript
Provide this ZIP as supplementary replication material. The public repository provides core experiment code and saved results; the selected JSONL sample and exploratory fragmentation files are supplied here. No new inference, news-text experiment, or independent human-labeling study is included. Keep this package with the submitted manuscript version. Code reuse remains subject to the author's licensing decision and upstream data attribution requirements.
