# Compact MSGT and prediction fusion: research materials

This repository accompanies the manuscript *Compact Multi-Scale Graph–Transformer with Source–Adapted Prediction Fusion for Rehabilitation Movement Assessment under Limited Target Supervision* (IEEE Access submission draft).

## Contents

- `results/data/`: all released aggregate tables, including the five-setting analysis, the matched U2K/IRDS comparison, model costs, diagnostics, and clearly marked post-hoc exploratory analyses. `results/MANIFEST.json` records row counts, columns, and source hashes; `results/CLAIMS_INDEX.json` records key descriptive counts.
- `predictions/fair_u2k_irds/`: 288 outer-test prediction CSV files for the U2K/IRDS matched MSGT and RI-GCN comparison. Each file records a model, dataset, outer split, target-participant budget, seed, and adaptation method. Columns include target labels and source, direct, fixed-half, and dynamic-fusion predictions.
- `splits/fair_u2k_irds_roles.json`: participant-disjoint source, adaptation, inner-validation, and outer-test roles for 24 comparison cells. Local machine paths were removed.
- `code/fair_adaptation/` and `code/dependencies/`: frozen scripts and model definitions used for the matched comparison. The RI-GCN upstream files retain their MIT license and attribution.

The original movement datasets are obtained from their creators, not redistributed here: KIMORE (doi:10.1109/TNSRE.2019.2923060), UI-PRMD (doi:10.3390/data3010002), IRDS (doi:10.3390/data6050046; data doi:10.5281/zenodo.4610859), KERAAL (https://keraal.enstb.org/KeraalDataset.html), and REHAB24-6 (doi:10.5281/zenodo.13305826). Follow each source's access and reuse terms.

## Integrity and scope

Run `python3 results/validate.py` to check the aggregate tables, hashes, columns, and key descriptive counts. `SHA256SUMS` covers every released file except itself.

The training scripts are archived research code. They require the original datasets, preprocessing inputs, prepared feature arrays, and source checkpoints used in the study; those materials are not included in this release. The CSV predictions and summary tables permit checking the reported comparisons without rerunning training. The 25/28, 17/18, and 24/24 counts summarize matched settings and should not be interpreted as independent-participant significance tests. Post-hoc exploratory weights are identified separately from the prespecified main rule.

No raw recordings, RGB video, original skeleton data, model weights, local machine paths, or personal contact information are included.
