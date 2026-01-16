# SIDA / Deepfake Forensics — Project README

Overview
- Collection of notebooks for training, fine-tuning and running a Vision-Transformer based deepfake / tampering detector.
- Focus areas: dataset prep, ViT training (timm), Hugging Face fine-tuning, attention-based localization, Grad-CAM/Grad-CAM++ explainability, and ONNX mobile inference.

Architecture (High-level)
- Data layer
  - ImageFolder / custom Dataset wrappers load images and labels.
  - Optional filtering to remove mask folders and stratified splits for train/val/test.
  - Label remapping to match model id ordering (e.g., ["synthetic","tampered","real"]).

- Preprocessing
  - Hugging Face AutoImageProcessor for HF models.
  - torchvision transforms for timm training (Resize/CenterCrop/RandomResizedCrop, Normalize).
  - Pixel tensors passed as `pixel_values` for HF models; torch tensors for timm models.

- Models
  - Hugging Face pipeline: AutoModelForImageClassification (SigLIP-like / HF ViT variants).
    - Config adjustments used at load: `config.output_attentions = True`, `attn_implementation="eager"` to force attention outputs.
    - Processor + model saved together for inference.
  - timm pipeline: timm.create_model('vit_base_patch16_224', ...) for CASIA training; classifier head replaced for binary detection.
  - ONNX/mobile: quantized or standard ONNX file (model_quantized.onnx / model.onnx) loaded via onnxruntime for mobile inference.

- Training
  - Hugging Face Trainer used in `deepfakefinetune.ipynb`.
    - Data collator, EarlyStoppingCallback, compute_metrics (accuracy), TrainingArguments (save/load best model).
  - Custom timm training loop in `Deepfake.ipynb`.
    - AdamW optimizer, CosineAnnealingLR, mixed precision (GradScaler), checkpoint saving when F1 improves.

- Inference / Explainability
  - Attention-based heatmaps:
    - Extract `outputs.attentions` (list of tensors) and average heads.
    - Handle token layouts: detect CLS token vs pure patch grid and reshape accordingly.
    - Resize patch-grid attention to original image, gamma-correct and colorize with OpenCV.
    - Common fixes: set `attn_implementation="eager"` and `config.output_attentions=True`.
  - Grad-CAM / Grad-CAM++:
    - Use `pytorch_grad_cam` for HF models; wrapper classes return .logits to satisfy Grad-CAM.
    - `reshape_transform` converts sequence tokens to (C,H,W) grid; target_layers selected dynamically (tries multiple attribute paths).
    - For timm ViT, use forward/backward hooks on attention or projection modules to compute Grad-CAM manually.
  - Manual Grad-CAM math:
    - Hooks capture activations & gradients; weights = global avg pool of gradients; heatmap = ReLU(sum(weights * activations)) then normalize & resize.

Per-notebook architecture summary
- deepfakefinetune.ipynb
  - Sections: load imagefolder dataset → remap folder labels to required order → define transforms using AutoImageProcessor stats → set train/val transforms → load HF model checkpoint with new num_labels and id2label/label2id → configure Trainer (with EarlyStopping) → run training → evaluation, metrics, plots → save & zip model for inference.
  - Includes helper `visualize_localization` that reloads saved model with eager attention and computes attention overlays for samples.

- Deepfake.ipynb
  - Sections: download CASIA dataset → filter masks → construct custom Dataset + DataLoader → create timm ViT model and replace head → training loop (AMP, optimizer, scheduler) with checkpointing → Grad-CAM hooks for timm ViT (forward/backward hooks capturing activations & gradients) → compute and visualize Grad-CAM for validation samples → save model weights.
  - Key patterns: hook-based Grad-CAM for timm, removal of CLS token when present, per-layer candidate testing for best visualization.