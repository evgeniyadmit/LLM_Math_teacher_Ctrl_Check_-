#!/bin/bash

# Create directory for the model
mkdir -p models/deepseek-math-7b

echo "⬇️ Downloading DeepSeek-Math 7B Instruct model from Hugging Face..."
huggingface-cli download \
  [deepseek-ai](chatgpt://generic-entity?number=0)/[deepseek-math-7b-instruct](chatgpt://generic-entity?number=1) \
  --local-dir models/deepseek-math-7b \
  --local-dir-use-symlinks false

echo "✅ Model downloaded locally into ./models/deepseek-math-7b"
