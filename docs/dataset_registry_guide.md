# Dataset Registry Guide

Registry file: `eeg_datasets_for_labram.csv`

## Required columns

- paper name
- authors
- paper link
- data link
- format
- paradigm
- subjects number
- channels / rate
- paper description
- dataset description
- additional information
- dataset total size
- per subject size

## Inclusion rules

Included rows are relevant to color/sound classification and Labram fine-tuning.

Explicitly excluded from this registry:

- Sciortino & Kayser 2023
- Brozova et al. 2025
- CNSP/Lalor
- THINGS-EEG
- EAV

## Size computation method

Implemented in `scripts/size_inventory.py`.

- OpenNeuro:
  - total size from OpenNeuro GraphQL `latestSnapshot.size`
  - per-subject size is estimated as `total_size / subject_count` when file-level per-subject sizes are unavailable via API.
- OSF:
  - recursive file listing via OSF API for total bytes
  - per-subject only if subject-tagged file structure is detectable.
- Other hosts:
  - `N/A` until a programmatic file listing endpoint is available.
