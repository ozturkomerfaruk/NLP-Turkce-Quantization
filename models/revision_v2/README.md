# Model files for the manuscript experiment

[EuroLLM](eurollm/) and [Qwen](qwen/) each contain their pinned source metadata and the FP16, Q8 and Q4 configurations used in the manuscript. The repository includes the model settings, tokenizer files and weight indexes. The [reproduction commands](../../docs/reproducibility_v2.md#rebuild-the-data-and-models) download the source weights, rebuild the MLX weights and verify that the converted metadata matches these files.

The [EuroLLM manifest](../../docs/eurollm_revision_manifest.json) and [Qwen manifest](../../docs/qwen_revision_manifest.json) record the reference files and their SHA-256 values, including the locally reconstructed weights. The older `models/eurollm-*` directories document the [first 1,000-pair experiment](../../docs/baseline_audit_v2.md). See the [third-party notices](../../THIRD_PARTY_NOTICES.md) for the model assets’ license and provenance.
