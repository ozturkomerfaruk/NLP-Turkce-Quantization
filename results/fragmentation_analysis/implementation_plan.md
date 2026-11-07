# Fragmentation analysis implementation plan

Goal: Complete the seven requested exploratory analysis steps without changing previous experiments.

Architecture: One CPU-only script reads frozen JSONL files and the existing approximate cluster CSV, validates identities and decisions, constructs pair-level measures, fits penalized logistic models with SciPy, and exports reproducible results. No model loading or scoring.

1. Test word counting, duplicate/missing ID rejection, and numerical ridge optimization using synthetic cases.
2. Implement strict 1000-pair joins, provenance hashes and original transition-count checks.
3. Use Unicode alphanumeric words with internal apostrophes; retain punctuation tokens in the recorded numerator as requested. Mean good/bad ratios are primary. Save signed and absolute differences.
4. Define quartiles with empirical quantile boundaries, keeping equal values together; report unequal group sizes. Cross-tabulate categories against global quartiles and report within-category quartiles separately.
5. Fit ridge logistic regression (sum negative log likelihood + lambda/2 times slope squared, unpenalized intercept, lambda=1). Primary controls: absolute FP16 mean-logp margin, mean word length, category. Continuous predictors standardized. Report fragmentation OR per original-sample SD and per one token/word.
6. Use 2000 category-stratified pair bootstraps and 2000 category-stratified cluster bootstraps. Refit standardization each replicate, convert coefficients to original-sample units. Report failures; abort if any fit fails. Preserve equal category weights in cluster resampling. Sensitivities: lambda 0.1/10, token length, signed fragmentation difference, log margin, leave-one-category-out.
7. Produce CSV, JSON, PNG and Turkish research report; run tests, inspect figure, verify source hashes and output invariants. Do not start article writing.
