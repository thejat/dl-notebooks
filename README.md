# Deep Learning and Modern Applications

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thejat/dl-notebooks)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red.svg)](LICENSE)

> **Alternate title:** Elements of Modern AI Models
> **Covers:** Transformers, LLM Fine-tuning, Generative AI, and Deep RL

This repository contains material for a deep learning course (IDS 576) at UIC.

**Audience:** Enthusiastic business analysts with intermediate Python programming experience.

---

## 📚 Table of Contents

- [Quick Start](#-quick-start)
- [Course Structure](#-course-structure)
- [Examples](#-examples)
- [Prerequisites](#-prerequisites)
- [Syllabus](#-syllabus)

---

## 🚀 Quick Start

### Option 1: Google Colab
Click the "Open in Colab" badge on any notebook to run it directly in your browser with free GPU access.

### Option 2: Local/Cloud VM Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/thejat/dl-notebooks.git
   cd dl-notebooks
   ```

2. **Set up Miniconda (Linux/macOS)**
   ```bash
   ./miniconda_setup.sh
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```

---

## 📖 Course Structure

| Module | Topic | Materials |
|--------|-------|-----------|
| M01 | Review & ML Pipeline | [Slides](slides/M01_review.pdf) |
| M02 | Feed-Forward Networks | [Slides](slides/M02_feedforward.pdf) |
| M03 | CNNs & Transfer Learning | [Slides](slides/M03a_cnn_and_transfer.pdf), [LoRA](slides/M03b_LoRA.pdf), [PyTorch](slides/M03c_Pytorch.pdf) |
| M04 | Text & NLP | [Slides](slides/M04_text.pdf) |
| M05 | Recurrent Networks & Attention | [Slides](slides/M05a_recurrent.pdf), [Attention](slides/M05b_attention.pdf) |
| M06 | Transformers | [Slides](slides/M06a_transformers.pdf) |
| M07 | Unsupervised Learning (VAEs, Diffusion & GANs) | [Slides](slides/M07_unsupervised.pdf) |
| M08 | Repeated Decision Making & Bandits | [Slides](slides/M08_repeated_decision_making.pdf) |
| M09 | Reinforcement Learning | [Slides](slides/M09_reinforcement.pdf) |
| M10 | Deep Reinforcement Learning | [Slides](slides/M10_deep_reinforcement.pdf) |
---

## 📁 Examples

| Category | Notebooks |
|----------|-----------|
| **Python & PyTorch Basics** | [Python Review](examples/M01_basics/Python_Review_IDS576.ipynb), [PyTorch Prelims](examples/M01_basics/Torch_Prelims.ipynb) |
| **Feed-Forward Networks** | [Linear Classifier](examples/M02_feedforward/Linear_Classifier_Example.ipynb), [FFN Classifier](examples/M02_feedforward/FFN_Classifier_Example.ipynb) |
| **CNNs** | [CNN Classifier](examples/M03_cnn_transfer/ConvolutionalNet_Classifier_Example.ipynb), [t-SNE MNIST](examples/M03_cnn_transfer/TSNE_Embedding_Example_MNIST.ipynb) |
| **NLP & RNNs** | [RNN Sentiment](examples/M05_recurrent/Seq2Seq_RNN_Simple_Sentiment_Analysis.ipynb), [LSTM Sentiment](examples/M05_recurrent/Seq2Seq_LSTM_Simple_Sentiment_Analysis.ipynb) |
| **LLM Fine-tuning** | [Fine-tuning & LoRA](examples/M03_cnn_transfer/Transfer_Learning_Finetuing_LoRA_Example.ipynb) |
| **Reinforcement Learning** | [Q-Learning CliffWorld](examples/M09_reinforcement/q_learning_cliffworld.ipynb) |

---

## 📋 Prerequisites

- **Python** 3.10+ (intermediate level)
- **Libraries:** PyTorch, NumPy, Pandas, Matplotlib, Jupyter
- **Prior coursework:** Data mining (IDS 572) and machine learning (IDS 575) or equivalent

---

## 📝 Syllabus

Please see the [Syllabus](Syllabus.md) for:
- Full course schedule and dates
- Assignment details and deadlines
- Project requirements ([Project.md](Project.md))
- Grading breakdown
- Textbook recommendations

Additional resources:
- [Lecture Goals](LectureGoals.md) - Learning objectives and external resources for each lecture

---

## 📄 License

Copyright © 2021-2026 Theja Tulabandhula. All Rights Reserved.

See [LICENSE](LICENSE) for details.
