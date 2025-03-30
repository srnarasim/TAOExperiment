# TAO Experiment: Text Classification with Test-time Adaptation

## Overview

This repository contains implementations of Test-time Adaptation for Out-of-distribution detection (TAO) applied to product classification. The experiments demonstrate how to detect products from new, unseen categories during inference time using entropy-based uncertainty estimation with both BERT-based models and Large Language Models (LLMs).

## Features

- Text classification of product descriptions into predefined categories
- Test-time adaptation to identify potential new/unseen product categories
- Entropy-based uncertainty estimation for out-of-distribution detection
- LLM-based implementation with LoRA for efficient adaptation to new categories
- Continuous test-time adaptation with feedback loop

## Implementations

This repository contains two different implementations of TAO:

### 1. BERT-based Implementation

The original implementation uses a BERT-based model for text classification with entropy-based uncertainty detection.

### 2. LLM with LoRA Implementation

The advanced implementation uses a Large Language Model (LLM) with Low-Rank Adaptation (LoRA) for more efficient and effective test-time adaptation.

Key features of the LLM-LoRA approach:
- Fine-tuning a pre-trained LLM (e.g., LLaMA-2) on known product categories
- Using LoRA for parameter-efficient fine-tuning
- Implementing test-time adaptation to dynamically handle new categories
- Creating lightweight LoRA adapters for new categories as they emerge
- Continuous adaptation with uncertainty-based human feedback loop

## Requirements

The project requires the following dependencies:

```python
transformers>=4.50.3
datasets>=3.0.1
torch>=2.6.0
pandas>=2.2.3
numpy>=2.2.4
scikit-learn>=1.4.0
accelerate>=1.5.2
bitsandbytes>=0.45.4
peft>=0.9.0
trl>=0.7.10
sentencepiece>=0.2.0
protobuf>=4.25.3
```

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### BERT-based Implementation

The original experiment is contained in the [TAOExperiment_fixed_updated.ipynb](TAOExperiment_fixed_updated.ipynb) notebook. You can run it directly in Google Colab by clicking the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srnarasim/TAOExperiment/blob/main/TAOExperiment_fixed_updated.ipynb)

### LLM-LoRA Implementation

The advanced LLM-based implementation is available in [TAOExperiment_LLM_LoRA.ipynb](TAOExperiment_LLM_LoRA.ipynb). You can run it in Google Colab by clicking the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srnarasim/TAOExperiment/blob/main/TAOExperiment_LLM_LoRA.ipynb)

## How It Works

### BERT-based Approach

1. A BERT-based model is fine-tuned on a dataset of product descriptions with known categories
2. During inference, the model calculates the entropy of the prediction probabilities
3. If the entropy exceeds a threshold, the product is flagged as potentially belonging to a new category

### LLM-LoRA Approach

1. A pre-trained LLM (e.g., LLaMA-2) is fine-tuned on known product categories using LoRA
2. During inference, the model calculates entropy-based uncertainty metrics
3. Products with high uncertainty are flagged as potential new categories
4. When new categories are confirmed, a new LoRA adapter is trained specifically for these categories
5. The system implements continuous test-time adaptation with human feedback

## Dataset

The experiments use a sample dataset (`balanced_data.csv`) containing product descriptions across various categories including Electronics, Clothing, Furniture, Accessories, and Kitchen items. The LLM implementation also includes additional categories for testing adaptation capabilities.

## Results

Both implementations demonstrate the ability to:
- Accurately classify products from known categories
- Identify potentially new product categories based on prediction uncertainty

The LLM-LoRA approach additionally provides:
- More efficient adaptation to new categories
- Better handling of complex product descriptions
- A continuous learning framework for evolving product catalogs

## Future Work

Potential improvements for this project include:
- Implementing active learning to selectively query human experts for uncertain predictions
- Developing more sophisticated uncertainty estimation methods beyond entropy
- Exploring parameter-efficient fine-tuning methods beyond LoRA (e.g., QLoRA, IA3)
- Creating a multi-adapter approach to handle different domains separately

## License

See the [LICENSE](LICENSE) file for details.