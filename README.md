# TAO Experiment: Text Classification with Test-time Adaptation

## Overview

This repository contains an implementation of Test-time Adaptation for Out-of-distribution detection (TAO) applied to product classification. The experiment demonstrates how to detect products from new, unseen categories during inference time using entropy-based uncertainty estimation.

## Features

- Text classification of product descriptions into predefined categories
- Test-time adaptation to identify potential new/unseen product categories
- Entropy-based uncertainty estimation for out-of-distribution detection

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
```

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

The main experiment is contained in the [TAOExperiment_fixed_updated.ipynb](TAOExperiment_fixed_updated.ipynb) notebook. You can run it directly in Google Colab by clicking the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srnarasim/TAOExperiment/blob/main/TAOExperiment_fixed_updated.ipynb)

## How It Works

1. A BERT-based model is fine-tuned on a dataset of product descriptions with known categories
2. During inference, the model calculates the entropy of the prediction probabilities
3. If the entropy exceeds a threshold, the product is flagged as potentially belonging to a new category

## Dataset

The experiment uses a sample dataset (`balanced_data.csv`) containing product descriptions across various categories including Electronics, Clothing, Furniture, Accessories, and Kitchen items.

## Results

The model demonstrates the ability to:
- Accurately classify products from known categories
- Identify potentially new product categories based on prediction uncertainty

## License

See the [LICENSE](LICENSE) file for details.