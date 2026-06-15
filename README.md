# Document Forgery Detection System

## Overview
This repository implements a Document Forgery Detection System using Error Level Analysis (ELA) and deep learning (EfficientNet). It includes training scripts, pretrained models, a Flask web app for inference, and utilities for reporting accuracy.

## Features
- ELA preprocessing for image artifacts
- Fine-tuned EfficientNet models for forgery detection
- Training and evaluation scripts
- Simple web interface for inference

## Quick Start
1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the web app (Flask):

```
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Training
- Prepare dataset under the `dataset/` folder with `fake/` and `real/` subfolders.
- Run training (example):

```
python train_Model.py
```

Training outputs and checkpoints are saved in `outputs/` (models and history files).

## Evaluation
- To evaluate or generate reports, see `report_accuracy.py`.
- Test results are stored in `outputs/test_results.npz` and a human-readable `outputs/report.txt`.

## Project Structure
- [app.py](app.py) — Flask web application for inference
- [train_Model.py](train_Model.py) — Training script
- [report_accuracy.py](report_accuracy.py) — Evaluation/reporting utilities
- [requirements.txt](requirements.txt) — Python dependencies
- [dataset/](dataset/) — Raw dataset: `fake/` and `real/`
- [outputs/](outputs/) — Trained models, histories, and reports
- [static/](static/) and [templates/](templates/) — Web UI assets and pages

## Models and Outputs
Pretrained and finetuned models in `outputs/`:
- `ela_efficientnet_pretrain_casia.h5`
- `ela_efficientnet_finetune_docs.h5`
- `ela_efficientnet_final_finetuned.h5`
- Training history: `pretrain_casia_history.npz`, `finetune_docs_history.npz`

## Notes and Tips
- The project uses ELA preprocessing — ensure input images are suitable for ELA (lossy formats like JPEG).
- If you add large model files, consider using Git LFS.

## License & Contact
----------------------------------------
