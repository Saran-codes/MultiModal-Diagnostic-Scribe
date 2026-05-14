# MultiModal Diagnostic Scribe

> Research project in computational cytopathology.

An end-to-end system that takes a raw Pap smear image and writes a structured, clinically interpretable Bethesda System report in JSON. The name reflects what the system does: it reads cytology slides and produces the written diagnostic record a pathologist would author — the scribe for the microscope.

The core technical challenge is that no public dataset pairs cervical cytology images with structured diagnostic reports. This project builds that dataset from scratch using a purpose-built synthetic report pipeline, then trains a Vision-Language Model on it.

---

## What the Scribe Produces

Given a Pap smear image, the system outputs:

```json
{
  "specimen_type": "Conventional Pap smear",
  "specimen_adequacy": "Satisfactory for evaluation",
  "general_categorization": "Epithelial cell abnormality",
  "microscopic_description": "Intermediate squamous cells with enlarged hyperchromatic nuclei, irregular nuclear membranes, and increased nuclear-to-cytoplasmic ratio. Koilocytic changes noted.",
  "interpretation": "High-grade squamous intraepithelial lesion (HSIL)",
  "recommendation": "Colposcopy with directed biopsy recommended."
}
```

Every field maps to a Bethesda System reporting category. The output is not a class label — it is actionable clinical language grounded in what the model sees.

---

## System Architecture

```
                     ┌─────────────────────────┐
                     │       Input Image        │
                     │      (1024 × 1024)       │
                     └────────────┬─────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │                                   │
                ▼                                   ▼
     ┌──────────────────────┐         ┌────────────────────────────┐
     │  Stage 1             │         │  Stage 2                   │
     │  Classifier          │         │  Vision Encoder            │
     │  ConvNeXt-Tiny       │         │  ResNet-50 (stripped)      │
     └──────────┬───────────┘         └─────────────┬──────────────┘
                │                                   │
                ▼                                   ▼
     ┌──────────────────────┐         ┌────────────────────────────┐
     │  Predicted Label     │         │  Adaptive Avg Pool         │
     │  NILM / LSIL / HSIL  ├────────▶│  16×16 → 256 Visual Tokens │
     │  ADENO / SCC         │  prompt │  MLP Bridge 2048→1024      │
     └──────────────────────┘  inject └─────────────┬──────────────┘
                                                     │
                                                     ▼
                                      ┌────────────────────────────┐
                                      │  BioGPT Decoder            │
                                      │  350M params               │
                                      │  1024 token context        │
                                      └─────────────┬──────────────┘
                                                     │
                                                     ▼
                                      ┌────────────────────────────┐
                                      │  Bethesda Report (JSON)    │
                                      │  specimen_type             │
                                      │  specimen_adequacy         │
                                      │  general_categorization    │
                                      │  microscopic_description   │
                                      │  interpretation            │
                                      │  recommendation            │
                                      └────────────────────────────┘
```

The Stage 1 predicted label is injected as natural language into the Stage 2 prompt before generation, grounding the language model in the classifier's output and eliminating clinical hallucination.

---

## Synthetic Report Pipeline

The defining challenge of this project: **no public dataset pairs cytology images with structured Bethesda reports.** Without training data, there is no scribe. The pipeline builds that data from scratch.

### How It Works

1. For each image, the ground-truth diagnostic label (NILM, LSIL, HSIL, ADENO, or SCC) is extracted from the dataset registry.
2. Two independent Bethesda-format JSON reports are generated per image using the Gemini 3 API, anchored by the ground-truth label to prevent hallucination.
3. A multi-worker orchestration layer (`manager.py`) runs parallel API workers with automatic sync and graceful stop, processing images at scale.
4. A diagnostic card generator (`bounding_box.py`) overlays cell bounding boxes and the microscopic description onto the source image — making each report visually traceable to the morphology it describes.

### Expert Validation

The generated reports were independently reviewed by a cytopathologist at AIIMS Deogarh:

| Bethesda Category | Expert Score (/ 5.0) |
|-------------------|----------------------|
| NILM              | 3.0                  |
| LSIL              | 3.1                  |
| HSIL              | 3.5                  |
| ADENO             | 3.0                  |
| SCC               | 3.2                  |
| **Mean**          | **3.16**             |

High-severity categories (HSIL, SCC) scored highest, which matters most clinically. The pipeline produced **over 9,000 report pairs** — the training corpus for Stage 2.

---

## Repository Structure

```
MultiModal-Diagnostic-Scribe/
│
├── classifier/
│   └── training/scripts/
│       ├── dataset.py          # Passive Scaling canvas pipeline, augmentation
│       ├── train.py            # ConvNeXt/ResNet training with Class-Weighted Focal Loss
│       └── grad_cam_viz.py     # Grad-CAM saliency map visualization
│
├── synthetic_report_pipeline/
│   └── scripts/
│       ├── generate_report.py  # Gemini 3 API worker: image → dual Bethesda JSON reports
│       ├── manager.py          # Multi-worker orchestration and sync loop
│       └── bounding_box.py     # Diagnostic card generator with cell bounding boxes
│
└── VLM/
    └── training/scripts/
        ├── model.py            # CytologyVLM: VisionEncoder + MLP bridge + BioGPT
        ├── dataset.py          # Stage 2 dataset: visual pipeline + multimodal tokenization
        ├── train.py            # 5-phase incremental training cascade
        ├── val.py              # Full evaluation: BLEU, ROUGE, METEOR, CIDEr
        ├── splits.py           # Stratified 80/10/10 dataset splitting
        └── attention_monitor.py # Attention forensics and spatial probe (W&B)
```

---

## Key Technical Decisions

**Passive Scaling** — Images from disparate sources (whole-slide vs. single-cell crops) vary enormously in dimension. Rather than stretching to 224×224, each image is placed on a 1024×1024 black canvas at its native aspect ratio. The nuclear-to-cytoplasmic (N/C) ratio is the primary morphological indicator of dysplasia; distorting it is clinically invalid.

**Class-Weighted Focal Loss** — The dataset is heavily imbalanced: benign NILM samples far outnumber rare malignant classes like SCC. Standard cross-entropy collapses to a majority-class predictor. Focal Loss (γ=2.0) with inverse-frequency class weights forces the optimizer to keep attending to hard, rare cases.

**Label Conditioning with 50% Masking** — The Stage 1 label is injected as a natural language prompt into BioGPT. To prevent posterior collapse (the model ignoring the image and parroting the prompt), 50% of training samples replace the guided prompt with a generic instruction, forcing the model to derive pathology from the 256 visual tokens.

**5-Phase Incremental Training** — Training a randomly initialized MLP bridge alongside frozen pre-trained models would corrupt BioGPT's biomedical weights with large early gradients. The cascade progressively unfreezes: MLP only → deeper MLP → higher visual resolution → top BioGPT layers → full end-to-end with ResNet layer4.

---

## Results

### Stage 1 — Classification (Test Set, 968 samples)

| Metric            | ResNet-50 | ConvNeXt-Tiny |
|-------------------|-----------|---------------|
| Global Accuracy   | 92.56%    | **92.67%**    |
| Balanced Accuracy | 89.00%    | **91.06%**    |
| Macro F1-Score    | 88.16%    | **89.72%**    |
| SCC Recall        | 0.78      | **0.91**      |

The SCC recall improvement (+13%) is the critical number. Missing a Squamous Cell Carcinoma is not a statistical error — it is a catastrophic clinical failure.

### Stage 2 — Report Generation (Test Set)

| Metric  | Score  |
|---------|--------|
| BLEU-4  | 50.44  |
| ROUGE-L | 28.00  |

Attention forensics confirmed the model is visually grounded: when generating terms like *hyperchromatic nuclei*, cross-attention weights concentrate on the actual abnormal cells in the image, not background debris.

---

## Data Sources

The harmonized dataset merges six sources: CPSMI, Herlev, SIPaKMeD, Mendeley cytology repository, BMT dataset, and image-caption pairs from *The Bethesda System for Reporting Cervical Cytology*. Final dataset: ~9,000 images split 80/10/10 (train/val/test), stratified by Bethesda category.

**Bethesda classes:** NILM · LSIL · HSIL · ADENO · SCC

---

## Hardware & Dependencies

Trained on an NVIDIA DGX Station (8× Tesla V100, 32 GB VRAM each) using Automatic Mixed Precision.

```
torch
torchvision
transformers          # BioGPT
google-genai          # Synthetic report pipeline
wandb                 # Training monitoring
evaluate              # ROUGE, BLEU, METEOR
nltk
pycocoevalcap         # CIDEr
pandas
Pillow
tqdm
seaborn
opencv-python
```
