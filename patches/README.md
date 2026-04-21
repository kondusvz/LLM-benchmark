# Upstream Patch Snapshots

This directory keeps the benchmark-specific changes that were applied to local
clones of upstream repositories during the dissertation project.

These patch files are included instead of committing the full cloned
repositories and their generated logs/data. They document the material changes
needed for the benchmark-integrated runs described in `THESIS_PATCH_NOTES.md`.

Files:

- `llmfe-efficiency-audit.patch`: local LLM-FE repo changes used for the
  efficiency audit path
- `featllm-multiclass.patch`: local FeatLLM repo changes used to make
  multiclass AUC evaluation robust on `nursery`

