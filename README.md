# ParaBERTa: Parallel Convolutional Projections of DeBERTa for Veracity Classification

**ParaBERTa** is a hybrid deep learning architecture that combines the contextual strength of **DeBERTa** with the local pattern recognition of **parallel CNNs** to detect fake news with high precision. This model is designed to capture both global semantic meaning and localized linguistic cues that indicate misinformation.

##  Overview

Online misinformation is spreading at an unprecedented rate. ParaBERTa aims to combat this by accurately classifying news articles as real or fake using a dual-strength approach:
- **DeBERTa** provides deep, context-aware representations of input text.
- **Parallel CNN layers** with varying kernel sizes (2, 3, 4) capture multiscale n-gram features, including biased phrases and clickbait patterns.

The output is then passed through convolutional and dense layers before classification via a **sigmoid function**.

---

## Performance Highlights

| Dataset | ParaBERTa Accuracy | Best Baseline | Improvement |
|--------|--------------------|----------------|-------------|
| ISOT   | 99.95%             | GPT-2 (99.93%) | +0.02%      |
| IFND   | 97.55%             | BERT (98.70%)  | +1.25%      |
| LIAR   | 68.27%             | RoBERTa (68.22%)| +0.05%     |

> ParaBERTa achieves **state-of-the-art accuracy** on major benchmark datasets.

---

## Key Features

- **DeBERTa V3 embeddings** for rich contextual representation
- **Parallel CNNs** for multiscale feature extraction
- Optimized with **ELU** and **ReLU** activations
- Trained and evaluated on **ISOT**, **IFND**, and **LIAR** datasets


## Datasets

- [ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/)
- [IFND Dataset](https://link.springer.com/article/10.1007/s40747-021-00552-1)
- [LIAR Dataset](https://aclanthology.org/P17-2067/)

---

## Tech Stack

- Python, PyTorch, HuggingFace Transformers
- DeBERTa V3, 1D CNN, ELU/ReLU activations
- Scikit-learn, NumPy, Pandas

