# Module Goals

## M01 : Motivating Applications, Machine Learning Pipeline  (Data, Models, Loss, Optimization), Backpropagation

 - [Online Demo](https://teachablemachine.withgoogle.com/)
 - [Micrograd code for backprop](https://github.com/karpathy/micrograd/tree/master), [engine.py](https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py)
 - [Colab file IO examples](https://colab.research.google.com/notebooks/io.ipynb)

### Goals

- Understand the key components to set up a classification task
- Relate business problems to machine learning methods
- Understand how chain rule works
- Understand why multiclass logistic regression may not work well even for 2D data

## M02 : Feedforward Networks: Nonlinearities, Convolutional Neural Networks: Convolution, Pooling

 - [Notebook: Pytorch Basics](https://github.com/sotte/pytorch_tutorial)
 - [Nonlinearities visualization](https://playground.tensorflow.org)
 - [CNN forward pass visualization](https://adamharley.com/nn_vis/cnn/2d.html)
 - [NN architecture visualization](https://github.com/lutzroeder/netron)

### Goals

- Get acquainted with the basics of Python
- Understand the notion of hidden layers and nonlinearities
- Convolution layer as collection of filters applied to input tensors
- Why pooling helps in reducing parameters downstream

## M03 : Jumpstarting Convolutional Neural Networks: Visualization, Transfer, Practical Models (VGG, ResNet)

 - [Overfitting and Dropout example](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit)
 - [Explaniability via Captum](https://captum.ai/docs/captum_insights)
 - Emerging Topics:
    - [Text to Video](https://lumiere-video.github.io/)
    - [Vision Models](https://omnihuman-lab.github.io/)
    - [Deepseek-V3](https://arxiv.org/pdf/2412.19437v1), [Deekseek-R1](https://arxiv.org/abs/2501.12948), [Deepseek-Janus-Pro](https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf) and related links: [1](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1), [2](https://www.interconnects.ai/p/deepseek-r1-recipe-for-o1), [3](https://github.com/deepseek-ai/) and [4](https://www.datacamp.com/tutorial/fine-tuning-deepseek-r1-reasoning-model).
 - Project Ideas from Meta ELI5 series: [1](https://www.youtube.com/playlist?list=PLzIwronG0sE7cCot-0yCbnuR6sAe7PIYX), [2](https://www.youtube.com/playlist?list=PLzIwronG0sE5ZVygDBFsV6EfdzyQ3u2m-) and [3](https://www.youtube.com/playlist?list=PLzIwronG0sE49zLk608yB5GKKT8WbEhnZ).
 - [LoRA](https://docs.google.com/presentation/d/1IxVFpmhX93oTVKfVtDMX_r1LAmXxolS1JXSkLORY7Y8/edit?usp=sharing)
 - [Comments on GPU Specs for Training](https://docs.google.com/presentation/d/1Smk_zMG7_XWsljpn9brURCm52HHVcSkD3CvA5ajTxNA/edit?usp=sharing)
 - Ethics
   - [Data Leak](https://dataportraits.org/)

### Goals

- Understand how to transfer parameters previously learned for a new task
- Know the different ways to debug a deep network
- Be aware of the different engineering tricks such as dropout, batch normalization
- Learn why image datasets can be enhanced using data augmentation
- Understand parameter-efficient fine-tuning techniques (LoRA, adapters) for pretrained models

## M04 : Text and Embeddings: Introduction to NLP, Word Embeddings, Word2Vec

 - [Spacy](https://spacy.io/usage/spacy-101)
 - Additional Reading:
   - [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)
   - CNN for sentence classification tasks. [link1](https://arxiv.org/pdf/1408.5882.pdf) and [link2](https://arxiv.org/pdf/1510.03820v4.pdf)
   - [Pytorch tutorial on using CNN for sentence classification: notebook 4](https://github.com/bentrevett/pytorch-sentiment-analysis)
 - Emerging Topics:
   - [Notebook LM by Google](https://notebooklm.google/)
   - [Controllable character video synthesis](https://menyifang.github.io/projects/MIMO/index.html)
   - [Small Vision Models](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/?_fb_noscript=1)
 - Ethics
   - [EU AI Act](https://artificialintelligenceact.eu/article/6/)
   - [OpenAI o1 System Card](https://openai.com/index/openai-o1-system-card/)

### Goals

- Understand how natural language elements (such as words) are processed in an analytics workflow
- Understand the shortcomings of methods such as Naive Bayes, Latent Dirichlet Allocation
- Realize that a CNN can also be used for a NLP task (sentence classification/sentiment analysis)
- What is word2vec and how does it help in NLP tasks?

## M05 : Recurrent Neural Networks and Transformers: Sequence to Sequence Learning, RNNs and LSTMs

 - [RNN example in Pytorch](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
 - [RNN function implementation in Pytorch](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
 - [LSTM example in Pytorch](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)

### Goals

- Know when prediction tasks can have sequential dependencies
- The RNN architecture and unfolding
- Know how LSTMs work
- Applications of 'sequential to sequential' models

## M06 : Advanced NLP: Attention, BERT and Transformers, LLMs, VLMs, MLLMs, Diffusion Models

 - Attention
   - [Attention Is All You Need](https://arxiv.org/abs/1706.03762) and an annotated [version](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
   - [Attention](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)
   - [Transformer Tutorial in Pytorch](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
   - [Transformer Visualization](https://poloclub.github.io/transformer-explainer/)
 - Models before GPT3
   - [BERT](https://arxiv.org/abs/1810.04805) and its [repository](https://github.com/google-research/bert)
   - [Illustrated BERT](https://jalammar.github.io/illustrated-bert/)
   - [GPT2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
   - [GPT3 Repository](https://github.com/openai/gpt-3)
 - Ethics
   - [Paper (2024): Decline of AI Data Commons](https://arxiv.org/abs/2407.14933)
 - Code
   - [SOTA Transformer Implementations including BERT and DistillBERT](https://github.com/huggingface/transformers) (for example [BERT-base-uncased](https://huggingface.co/bert-base-uncased))
   - [Open Deep Research Model](https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research)
   - [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
   - [Unsloth: Fast LLM Fine-tuning](https://github.com/unslothai/unsloth)
 - Papers
   - [Llama](https://arxiv.org/abs/2302.13971), [Llama 2](https://arxiv.org/abs/2307.09288), [Llama 3](https://arxiv.org/abs/2407.21783)

### Goals

- Be able to explain self-attention and how it differs from simpler attention mechanisms seen in sequence to sequence models
- Be able to reason about keys, values and queries in self-attention
- Be able to recall the key characteristics of BERT and how pre-trained models can be used for NLP tasks.
- Understand the architecture and training paradigm of Large Language Models (LLMs)
- Know the basics of LLM fine-tuning using parameter-efficient methods (LoRA, PEFT)
- Be aware of vision-language models (VLMs) and multimodal LLMs (MLLMs)

## M07 : Unsupervised Deep Learning: Variational Autoencoders, Diffusion Models, Generative Adversarial Networks

 - [VAE in Pytorch](https://github.com/pytorch/examples/tree/master/vae) from [Pytorch examples repository](https://github.com/pytorch/examples)
 - Diffusion Models
    - [Introduction](https://github.com/huggingface/diffusion-models-class/tree/main/unit1)
    - [Slides](https://deeplearning.cs.cmu.edu/S25/document/slides/lec23.diffusion_s25.pdf)
 - [Notebook: GAN example on CelebFaces Attributes (CelebA) Dataset](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) ([dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html))
 - [VAE in Pytorch](https://github.com/pytorch/examples/tree/master/vae) from [Pytorch examples repository](https://github.com/pytorch/examples)
- [GAN Demo by Google 2020](https://ai.googleblog.com/2020/11/using-gans-to-create-fantastical.html)
- [GAN animation](https://poloclub.github.io/ganlab/)

### Goals

- Meaning of generative modeling
- What are variational autoencoders (VAEs) and where can they be used?
- The intuition behind generative adversarial networks (GANs)
- Differences between GANs and VAEs

## M08 : Online Learning: A/B Testing, Multi-armed Bandits, Contextual Bandits

 - [Bandit Implementations in Python: SMPyBandits](https://smpybandits.github.io/)
 - [A blog post on Bandits by Lilian Weng](https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html)

### Goals

- What is online learning? How is it different from supervised learning?
- Relation between forecasting and decision making
- The multi armed bandit problem and solutions
- Contextual bandits

## M09 : Reinforcement Learning: Policies, State-Action Value Functions, Bellman Equations, Q Learning

 - [Openai Gym](https://github.com/openai/gym)
 - [RL in Pytorch](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) from [Pytorch examples repository](https://github.com/pytorch/examples)
 - [RL with Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)
 - [RLHF for ChatGPT like Assistants](https://arxiv.org/pdf/2204.05862.pdf)
 - [Implementations of RL](https://github.com/dennybritz/reinforcement-learning)
 - [Flappy Bird with Q learning](http://sarvagyavaish.github.io/FlappyBirdRL/)
 - [ML Katas: Cliffworld with Q learning](https://www.bpesquet.fr/mlkatas/coding/q_learning_cliffworld.html)

### Goals

- What is reinforcement learning?
- Basics of Markov Decision Processes
- Policies, Value functions and how to think about these two objects
- Be able to understand the difference between Bellman Expectation Equation and Bellman Optimality Equation
- Intuitive reasoning for the Q-Learning update rule
- Be able to identify relationships between state value functions, state-action value functions and policies

## M10 : Deep Reinforcement Learning: Function Approximation, DQN for Atari Games,  DQN for Atari Games, MCTS for AlphaGo

 - [OpenAI Baselines for RL](https://github.com/openai/baselines)
 - [Cartpole environment with DQN](https://github.com/seungeunrho/minimalRL/blob/master/dqn.py)

### Goals

- Know the role of function approximation in Q-learning
- Be able to understand the key innovations in the DQN model 
- Identify the differences between Monte Carlo tree search vs Monte Carlo rollouts
- Be able to identify key compoments of the AlphaGo (and variants such as AlphaZero) Go playing agent

## M11 : AI Ethics, Fairness, Accountability, Transparency and Sustainability

 - Textbooks (see [Syllabus](Syllabus.md))
   - [Responsible AI](https://learning.oreilly.com/library/view/responsible-ai/9781098102425/) by Patrick Hall and Rumman Chowdhury (2022)
   - [Practical Fairness](https://learning.oreilly.com/library/view/practical-fairness/9781492075721/) by Aileen Nielsen (2020)
   - [Fairness and Machine Learning: Limitations and Opportunities](https://fairmlbook.org/) by Barocas, Hardt and Narayanan (2018)
   - [The Framework for ML Governance](https://learning.oreilly.com/library/view/the-framework-for/9781098100483/) by Kyle Gallatin (2021)
 - Fairness Tools and Libraries
   - [IBM AI Fairness 360](https://aif360.mybluemix.net/)
   - [Google What-If Tool](https://pair-code.github.io/what-if-tool/)
   - [Microsoft Fairlearn](https://fairlearn.org/)
 - Explainability and Transparency
   - [SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/)
   - [LIME (Local Interpretable Model-agnostic Explanations)](https://github.com/marcotcr/lime)
 - Sustainability
   - [ML CO2 Impact Calculator](https://mlco2.github.io/impact/)
   - [Green AI Paper](https://arxiv.org/abs/1907.10597)
 - Regulation and Guidelines
   - [EU AI Act](https://artificialintelligenceact.eu/)
   - [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)

### Goals

- Understand the key principles of responsible AI: fairness, accountability, transparency, and ethics
- Be able to identify sources of bias in ML pipelines (data, model, deployment)
- Know how to use fairness metrics and tools to evaluate and mitigate bias in models
- Understand the importance of model interpretability and explainability for stakeholder trust
- Be aware of the environmental impact of training large models and strategies for sustainable AI
- Familiarize with regulatory frameworks (EU AI Act, NIST) governing AI deployment