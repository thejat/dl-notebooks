# Deep Learning and Modern Applications

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thejat/dl-notebooks)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red.svg)](LICENSE)

> **Alternate title:** Elements of Modern AI Models

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

### Option 1: Google Colab (Recommended)
Click the "Open in Colab" badge on any notebook to run it directly in your browser with free GPU access.

### Option 2: Local Setup

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
| L01 | Backpropagation & ML Pipeline | [Slides](slides/lec01_backprop.pdf) |
| L02 | Feed-Forward Networks | [Slides](slides/lec02_ffn.pdf) |
| L03 | CNNs & Transfer Learning | [Slides](slides/lec03_cnns_transfer.pdf) |
| L04 | NLP & Word Embeddings | [Slides](slides/lec04_nlp.pdf) |
| L05 | Seq2Seq & Attention | [Slides](slides/lec05_seq2seq.pdf), [Slides](slides/lec05a_attention.pdf) |
| L06 | Advanced NLP & Transformers | [Slides](slides/lec06_advanced_text.pdf) |
| L07-08 | VAEs & GANs | [Slides](slides/lec07_08_VAEs_and_GANs.pdf) |
| L09 | Online Learning & Bandits | [Slides](slides/lec09_online_ml.pdf) |
| L10-11 | Intro to RL | [Slides](slides/lec10_11_intro2RL.pdf) |
| L12 | Deep RL | [Slides](slides/lec12_deepRL.pdf) |

---

## 📁 Examples

| Category | Notebooks |
|----------|-----------|
| **Python & PyTorch Basics** | [Python Review](examples/basics_of_python_and_pytorch/Python_Review_IDS576.ipynb), [PyTorch Prelims](examples/basics_of_python_and_pytorch/Torch_Prelims.ipynb) |
| **Feed-Forward Networks** | [Linear Classifier](examples/feed_forward_networks/Linear_Classifier_Example.ipynb), [FFN Classifier](examples/feed_forward_networks/FFN_Classifier_Example.ipynb) |
| **CNNs** | [CNN Classifier](examples/convolutional_neural_networks/ConvolutionalNet_Classifier_Example.ipynb), [t-SNE MNIST](examples/convolutional_neural_networks/TSNE_Embedding_Example_MNIST.ipynb) |
| **NLP & RNNs** | [RNN Sentiment](examples/recurrent_neural_networks/Seq2Seq_RNN_Simple_Sentiment_Analysis.ipynb), [LSTM Sentiment](examples/recurrent_neural_networks/Seq2Seq_LSTM_Simple_Sentiment_Analysis.ipynb) |
| **Transfer Learning** | [Fine-tuning & LoRA](examples/transfer_learning/Transfer_Learning_Funetuing_LoRA_Example.ipynb) |
| **Reinforcement Learning** | [Q-Learning CliffWorld](examples/reinforcement_learning/q_learning_cliffworld.ipynb) |

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
