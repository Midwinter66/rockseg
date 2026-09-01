# Auxiliary Scripts

This directory contains non-production helpers. The four current pipeline
entry points deliberately remain in the repository root:

- `run_rockseg.py`
- `run_3d_validation.py`
- `run_3d_validation_fast.py`
- `run_volume_estimation.py`

| Location | Contents | Status |
| --- | --- | --- |
| `document_tools/` | Word/manuscript formatting and revision helpers | Historical manuscript support |
| `presentations/2026-08-26/` | Scripts used to prepare the preserved 2026-08-26 progress presentation | Reproducibility support |
| `temporary/pdf_inspection/` | Small PDF extraction helpers and their generated JSON | Temporary; safe to delete when no longer needed |

Scripts here do not define the frozen DOM2 volume result or the frozen
Shape-Aware V2 model.
