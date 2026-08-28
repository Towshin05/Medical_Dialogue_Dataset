
# Bengali Medical LLM Post-Training with Qwen2.5

## Overview

This project documents my first hands-on experience with LLM post-training
during the Nascenia AI Hackathon, focused on Bengali medical dialogue
generation.

The goal was to fine-tune a transformer-based language model to generate
doctor-style responses to Bengali medical queries while respecting the
competition's under-3B-parameter constraint.

## Models

I experimented with:

- Qwen2.5-1.5B-Instruct
- Qwen2.5-3B-Instruct

## Dataset

The dataset contains Bengali medical patient-doctor dialogues.

Before training, I performed data cleaning and preprocessing, including:

- Removing unnecessary symbols and marks
- Removing icons and noisy characters
- Cleaning ambiguous or low-quality words
- Handling noisy mixed Bengali-English text
- Formatting patient/doctor conversations for training

The final training format was based on input/output dialogue pairs.

## Post-Training Approach

I used:

- Supervised Fine-Tuning (SFT)
- LoRA
- QLoRA
- 4-bit NF4 quantization

For the Qwen2.5-3B model, QLoRA allowed the model to be loaded in
4-bit precision while training a small set of LoRA parameters instead
of updating all model weights.

## Training Setup

- Base model: Qwen2.5-3B-Instruct
- Quantization: 4-bit NF4
- Fine-tuning: LoRA / QLoRA
- Sequence length: 2048
- Optimizer: Paged AdamW 8-bit
- Gradient accumulation: Used to reduce GPU memory requirements
- Gradient checkpointing: Enabled
- GPU: NVIDIA Tesla T4
- Frameworks:
  - PyTorch
  - Hugging Face Transformers
  - PEFT
  - TRL
  - bitsandbytes



