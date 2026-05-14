# MultiModal Diagnostic Scribe

> Research project in computational cytopathology.

An end-to-end two-stage deep learning system that takes a raw Pap smear image and outputs a structured, clinically interpretable Bethesda System report in JSON. The system moves beyond categorical classification by combining a high-resolution vision classifier with a causal Vision-Language Model (VLM), giving a diagnosis a clinical voice.

---

## The Problem

Automated cytology classifiers produce a single integer. A class label gives a clinician no morphological reasoning and no actionable clinical language. Compounding this, no public dataset pairs cervical cytology images with structured diagnostic reports, which has blocked serious work on explainable text generation for this domain.

This project addresses both problems directly: it builds the multimodal architecture and generates the training data from scratch.

---

## System Architecture

```
                        ┌─────────────────────────────────────┐
                        │           Input Image               │
                        │         (1024 × 1024)               │
                        └──────────────┬──────────────────────┘
                                       │
                   ┌───────────────────┴───────────────────┐
                   │                                       │
                   ▼                                       ▼
     ┌─────────────────────────┐           ┌──────────────────────────┐
     │   Stage 1: Classifier   │           │  Stage 2: Vision Encoder │
     │  ConvNeXt-Tiny / ResNet │           │  ResNet-50 (stripped)    │
     └───────────┬─────────────┘           └──────────────┬───────────┘
                 │                                        │
                 ▼                                        ▼
     ┌───────────────────────┐           ┌──────────────────────────────┐
     │   Classification Head │           │  Adaptive Avg Pool → 16×16   │
     │   (5-class Bethesda)  │           │  256 Visual Tokens           │
     └───────────┬───────────┘           └──────────────┬───────────────┘
                 │                                       │
                 ▼                                       ▼
     ┌───────────────────────┐           ┌──────────────────────────────┐
     │   Predicted Label     │           │  Cross-Modal MLP Bridge      │
     │  NILM / LSIL / HSIL  ├──────────▶│  2048 → 1024 (GELU)          │
     │    ADENO / SCC        │  prompt   │  Visual Scaling (1.2×)       │
     └───────────────────────┘ inject   └──────────────┬───────────────┘
                                                        │
                                                        ▼
                                        ┌──────────────────────────────┐
                                        │   BioGPT Language Decoder    │
                                        │   (350M params, 1024 ctx)    │
                                        └──────────────┬───────────────┘
                                                        │
                                                        ▼
                                        ┌──────────────────────────────┐
                                        │      Bethesda JSON Report    │
                                        │  specimen_type               │
                                        │  specimen_adequacy           │
                                        │  general_categorization      │
                                        │  microscopic_description     │
                                        │  interpretation              │
                                        │  recommendation              │
                                        └──────────────────────────────┘
```

The Stage 1 predicted label is injected directly into the Stage 2 text prompt before generation. This grounds the language model in the classifier's output and eliminates clinical hallucinations.

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

## Key Design Decisions

**Passive Scaling** — Images from disparate sources (whole-slide vs. single-cell crops) vary enormously in dimension. Rather than stretching everything to 224×224 (which destroys the nuclear-to-cytoplasmic ratio), each image is scaled to fit within a 1024×1024 black canvas while preserving its aspect ratio. The N/C ratio is the primary morphological indicator of dysplasia; distorting it is clinically invalid.

**Class-Weighted Focal Loss** — The dataset is heavily imbalanced: benign NILM samples far outnumber rare malignant classes like SCC. Standard cross-entropy collapses to a majority-class predictor. Focal Loss with inverse-frequency class weights forces the optimizer to keep paying attention to hard, rare cases.

**Label Conditioning with 50% Masking** — The Stage 1 label is injected as a natural language prompt into BioGPT. To prevent posterior collapse (the model ignoring the image and just parroting the prompt), 50% of training samples replace the guided prompt with a generic instruction, forcing the model to derive the pathology from the 256 visual tokens.

**Synthetic Ground Truth Pipeline** — No public dataset pairs cytology images with structured reports. The Gemini 3 API was used to generate two independent Bethesda-format JSON reports per image, anchored by the ground-truth diagnostic label to prevent hallucination. The pipeline produced over 9,000 report pairs, independently reviewed by a cytopathologist at AIIMS Deogarh (mean expert score: 3.16/5.0, with high-severity categories HSIL and SCC scoring 3.5 and 3.2 respectively).

**5-Phase Incremental Training** — Training a randomly initialized MLP bridge alongside frozen pre-trained vision and language models would corrupt BioGPT's biomedical weights with large early gradients. The training cascade progressively unfreezes: MLP only → deeper MLP → higher visual resolution → top BioGPT layers → full end-to-end with ResNet layer4.

---

## Results

### Stage 1 — Classification (Test Set, 968 samples)

| Metric            | ResNet-50 | ConvNeXt-Tiny |
|-------------------|-----------|---------------|
| Global Accuracy   | 92.56%    | **92.67%**    |
| Balanced Accuracy | 89.00%    | **91.06%**    |
| Macro F1-Score    | 88.16%    | **89.72%**    |
| SCC Recall        | 0.78      | **0.91**      |

The SCC recall improvement (+13%) is the critical number. Missing a Squamous Cell Carcinoma is not a statistical error; it is a catastrophic clinical failure.

### Stage 2 — Report Generation (Test Set)

| Metric  | Score  |
|---------|--------|
| BLEU-4  | 50.44  |
| ROUGE-L | 28.00  |
| METEOR  | —      |

Attention forensics confirmed the model is visually grounded: when generating terms like *hyperchromatic nuclei*, cross-attention weights concentrate on the actual abnormal cells in the 1024×1024 image, not background debris.

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
pycocoevalcap         # CIDEr (optional)
pandas
Pillow
tqdm
seaborn
opencv-python
```
