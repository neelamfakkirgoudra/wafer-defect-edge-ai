# wafer-defect-edge-ai
# Edge AI – Wafer Defect Classification

## Problem Statement
Develop an Edge-AI capable system to automatically detect and classify defects
in semiconductor wafer images using AI/ML techniques.

## Dataset
- Total Classes: 8 (Clean + 7 defect types)
- Images are grayscale SEM wafer images
- Dataset split:
  - Train
  - Validation
  - Test

## Model
- Architecture: MobileNetV2 (Transfer Learning)
- Framework: PyTorch
- Input Size: 224x224
- Output: 8 classes
- Export Format: ONNX

## Training
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Epochs: 25
- Batch Size: 16
- Learning Rate: 0.0001
- Platform: CPU/GPU (based on availability)

## Evaluation (Test Set)
- Accuracy: 25%
- Precision: 26.92%
- Recall: 25%
- Confusion Matrix: Included in results/

## Edge Deployment
- Model exported to ONNX format
- Ready for NXP eIQ toolchain
- Model size: ~268 KB

## Files
- `code/` – Training, evaluation & export scripts
- `model/` – ONNX model
- `results/` – Metrics and confusion matrix
- Sem Images Dataset:-
- Google Drive Folder:https://drive.google.com/drive/folders/1Z6a9h0ogrxXXV5VEk84Vcl_jFVsiESO1?usp=sharing
- Resubmitted Google Drive Link: https://drive.google.com/drive/folders/1kUXwZZXjfbHESFb-h6ItFqS4bFFQNfDd?usp=sharing


## Author
NEELA M F
