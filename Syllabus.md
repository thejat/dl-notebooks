# Deep Learning and Modern Applications

## Alternate title: Elements of Modern AI Models

## 1. Summary

- Term, CRN and Number of credits: Spring 2026, 38063 (IDS 576) and 4 resp.
- Instructor: [Dr. Theja Tulabandhula](https://theja.org)

## 2. Objective

The goal of this class is to cover a subset of advanced machine learning techniques, after students have seen the basics of data mining (such as in IDS 572) and machine learning (such as in IDS 575). 

Broadly, we will cover topics spanning **deep learning and reinforcement learning**. In particular, we will study popular deep learning architectures, their design choices and how they are trained. This will be motivated by business applications dealing with image, text and tabular data, including LLM-powered applications. Finally, we will look at online and reinforcement learning frameworks and their role in sequential decision making settings such as retail.

A tentative list of topics is as follows:

 - Backpropagation and feed forward neural networks
 - Convolutional networks (CNNs)
 - Recurrent networks and Long short-term memory networks (LSTMs)
 - Attention mechanism and transformers (BERT, GPT-3)
 - Variational autoencoders (VAEs)
 - Generative adversarial networks (GANs)
 - Large Language Models (LLMs): training, fine-tuning (LoRA, PEFT), inference and applications
 - Deep reinforcement learning and Deep Q-Network (DQN)
 - Deep learning design choices such as optimizers (Adam, RMSprop), nonlinearities, embeddings, attention, dropout, batch normalization etc.

We will also discuss aspects of **AI ethics, fairness, accountability, transparency and sustainability** in this course. ML systems are being adopted in a growing number of contexts, fueled by big data. These systems filter, sort, score, recommend, personalize, and otherwise shape human experience, increasingly making or informing decisions with major impact on access to, e.g., credit, insurance, healthcare, parole, social security, and immigration. Although these systems may bring myriad benefits, they also contain inherent risks, such as codifying and entrenching biases; reducing accountability, and hindering due process; they also increase the information asymmetry between individuals whose data feed into these systems and big players capable of inferring potentially relevant information. Under the *economic-social* intersection of sustainability (social: standard of living, education, community, equal opportunity; economic: profit, cost saving, economic growth), we will revisit how DL models and related ML systems could be designed that tackle the aforementioned challenges while providing value.

## 3. Course Logistics

 - Meeting Times: Wednesdays 6.00 PM to 8.30 PM
 - Location: Lecture Center Building A | Room A007
 - Staff: 
   - Instructor: [Dr. Theja Tulabandhula](https://theja.org)
     - Office Hours: Wednesdays 1-2pm by prior email appointments at UH 2407/Zoom/Teams
 - **Course Platform**: All course materials, announcements, group registration, and submissions will be managed through **[Blackboard](https://uic.blackboard.com/)**.

### 3.1. Tentative Dates

 - 01/14: lecture
 - 01/21: lecture
 - 01/27: assignment 1 due
 - 01/28: lecture
 - 02/04: lecture
 - 02/11: lecture
 - 02/17: assignment 2 due
 - 02/18: lecture
 - 02/25: lecture
 - 03/04: lecture
 - 03/10: assignment 3 due
 - 03/11: lecture
 - 03/17: project intermediate report, plan and code due
 - 03/18: lecture
 - 03/25: (no class: spring break)
 - 04/01: lecture
 - 04/08: lecture
 - 04/15: lecture
 - 04/22: lecture
 - 04/28: project final report+code due
 - 04/29: student project presentations

### 3.2. Tentative Modules

*See [this page](https://github.com/thejat/dl-notebooks/blob/master/ModuleGoals.md) for goals, module specific external links and [this folder](https://github.com/thejat/dl-notebooks/tree/master/slides) for the slides.*

 -  M01 : Motivating Applications, Machine Learning Pipeline  (Data, Models, Loss, Optimization), Backpropagation
 -  M02 : Feedforward Networks: Nonlinearities, Convolutional Neural Networks: Convolution, Pooling
 -  M03 : Transfer Learning and Fine-tuning: Visualization, LoRA, Practical Models (VGG, ResNet)
 -  M04 : Text and Embeddings: Introduction to NLP, Word Embeddings, Word2Vec
 -  M05 : Recurrent Neural Networks and Transformers: Sequence to Sequence Learning, RNNs and LSTMs
 -  M06 : Advanced NLP: Attention, BERT and Transformers, LLMs, VLMs, MLLMs, Diffusion Models
 -  M07 : Unsupervised Deep Learning: Variational Autoencoders, Generative Adversarial Networks
 -  M08 : Online Learning: A/B Testing, Multi-armed Bandits, Contextual Bandits
 -  M09 : Reinforcement Learning: Policies, State-Action Value Functions, Bellman Equations, Q Learning 
 -  M10 : Deep Reinforcement Learning: Function Approximation, DQN for Atari Games, MCTS for AlphaGo
 -  M11 : AI Ethics, Fairness, Accountability, Transparency and Sustainability

## 4. Textbook and Materials

### 4.1. Textbooks for DL and RL

 - [Deep Learning](https://www.deeplearningbook.org/) by Ian Goodfellow and Yoshua Bengio and Aaron Courville (2016).
 - [Dive into Deep Learning](https://d2l.ai/index.html) by Aston Zhang and Zachary C. Lipton and Mu Li and Alexander J. Smola (2020).
 - [Deep Learning for Coders with fastai and PyTorch: AI Applications Without a PhD](https://github.com/fastai/fastbook) by Jeremy Howard and Sylvain Gugger (2020). It is available on [Amazon]( https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527).
 - [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) by Richard S. Sutton and Andrew G. Barto (2018).

### 4.2 Textbooks for AI Ethics and Sustainability

 - [Book: "Responsible AI". 2022. by Patrick Hall, Rumman Chowdhury. O'Reilly Media, Inc.](https://learning.oreilly.com/library/view/responsible-ai/9781098102425/)
 - [Book: "Practical Fairness". 2020. By Aileen Nielsen. O'Reilly Media, Inc.](https://learning.oreilly.com/library/view/practical-fairness/9781492075721/)
 - [Book: "Fairness and machine learning: Limitations and Opportunities." Barocas, S., Hardt, M. and Narayanan, A., 2018.](https://fairmlbook.org/)
 - [Book: "The Framework for ML Governance" by Kyle Gallatin. 2021.  O'Reilly Media](https://learning.oreilly.com/library/view/the-framework-for/9781098100483/)

### 4.3. Software and Hardware

 - Any OS should be okay. If in doubt, run a virtual machine running linux (this will be discussed in the class). Some of the software we will work with are:
   - [Python](https://www.python.org/): Refer to [this set of notes](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-189-a-gentle-introduction-to-programming-using-python-january-iap-2011/lectures/) or [this](http://stanfordpython.com/) to get started in Python.
   - [Jupyter](https://jupyter.org/)
   - Pytorch, tensorflow, keras, matplotlib, numpy, scipy, pandas, gym, random, ...
   - ...
 - There will be varied computing resources needed for this course. We will rely on [Google Colab](https://colab.research.google.com/notebooks/), especially the GPU/TPU to train our deep learning models. 

## 5. Assignments and Project

You should form groups of _strictly_ 2 students for the assignment and project components. Reach out to your classmates early (and register your groups in the spreadsheet linked on the course platform).

### 5.1. Assignment 

The assignment links are below:

 1. [Assignment 1](https://github.com/thejat/dl-notebooks/blob/master/Assignment1.ipynb) 
 2. [Assignment 2](https://github.com/thejat/dl-notebooks/blob/master/Assignment2.ipynb) 
 3. [Assignment 3](https://github.com/thejat/dl-notebooks/blob/master/Assignment3.ipynb) 

These involve reimplementing recent deep-learning techniques and understanding their behavior on interesting datasets. Always mention any sources that were relied on in your assignment solutions. Submission deadline is BEFORE 11.59 PM on the concerned day. Late submissions will have an automatic 20% penalty per day without exceptions. Use the course platform for uploads.

### 5.2. Project

 - Students are expected to apply what they learn in the course and demonstrate their understanding by undertaking a suitable project. 
 - A preliminary documentation along with the scripts/codes/commands used is to be submitted midway and a final version of the same is to be submitted right before the end of the term (see schedule above and the course platform). 
- The scope, evaluation criteria used, examples and other details are available [here](https://github.com/thejat/dl-notebooks/blob/master/Project.md). 
- Submission deadline is BEFORE 11.59 PM on the concerned day. Late submissions will have an automatic 20% penalty per day. Use the course platform for uploading your work as a _single zip_ file.

## 6. Grades

 - Grades will be based on:
   - the assignments (3 times 10%),
   - the project (see project details above) (10% for intermediate and 50% for final), and
   - course participation (including but not limited to attendance, interaction with the instructor and the TA (the latter if applicable), and how well you support your group) (10%).

## 7. Generative AI Usage Policy

### Overview

Generative AI tools such as OpenAI ChatGPT, Microsoft Copilot (Bing Chat), Google Gemini, and similar systems can generate text, images, code, and other media in response to user prompts. While these tools may support learning, their use in this course is governed by the following policy to ensure academic integrity, responsible engagement, and alignment with University of Illinois System guidance.

Students should be aware that generative AI systems may produce outputs that appear plausible but are factually incorrect, fabricated, or internally inconsistent. This phenomenon, often referred to as hallucination, occurs because these systems do not inherently verify facts or ensure correctness. As a result, AI-generated outputs may:

* Infer relationships that do not exist
* Invent citations, names, references, or data
* Blend correct and incorrect information in misleading ways

Students are fully responsible for the accuracy, originality, and integrity of all submitted work, regardless of whether AI tools were used.

### General Guidelines

1. Use generative AI tools only in ways explicitly permitted by this syllabus or by assignment instructions.
2. Clearly document and disclose all uses of generative AI in coursework.
3. Independently verify the accuracy of all AI-assisted content, including facts, data, citations, and code.

### Permitted Uses of Generative AI

Unless otherwise specified for a particular assignment, students may use generative AI tools for the following purposes:

* Editing or shortening their own written text
* Revising their own writing for grammar, spelling, or clarity
* Creating personal study aids (for example, flashcards or summaries)
* Practicing explanations of course concepts or testing understanding
* Conducting preliminary or background research on course-related topics

AI tools may not be used to replace original thinking, analysis, or problem-solving unless explicitly authorized.

### Exams and Quizzes

* **Closed-book assessments:** Use of generative AI tools is not allowed.
* **Open-book assessments:** Use of generative AI tools is permitted only if explicitly stated in the assessment instructions. Conditions will be specified on a case-by-case basis.

Unauthorized use of AI during assessments constitutes a violation of academic integrity.

### Documentation Requirements

When generative AI tools are used, students must be able to document that use. Documentation may include:

* A record or journal of prompts and AI-generated responses
* A link to the relevant AI chat history, if available

Students must submit this documentation upon request.

### Citation and Attribution

* All use of generative AI must be properly cited and disclosed.
* AI tools should be cited following APA guidelines, including the text of the prompt when relevant.
* AI-generated content is not a valid primary source for factual claims. Students must locate, verify, and cite original authoritative sources for all facts, data, and claims.

Students are expected to follow the University of Illinois System guidance on generative AI use by students.

### Academic Integrity

Failure to comply with this policy constitutes a violation of academic integrity and will be handled in accordance with the UIC Student Code of Conduct.

Misuse of generative AI includes, but is not limited to:

* Submitting AI-generated work as one's own without disclosure
* Using AI tools to fabricate data, results, or sources
* Relying on AI systems in ways that circumvent learning objectives

### Data Privacy Notice

Generative AI platforms may collect, store, or share user data. Students should not input personal, confidential, proprietary, or sensitive information into these systems. Guidance on data security and privacy is available through UIC Information Technology resources.

### AI Analysis Disclaimer

Student submissions may be stored in anonymized repositories and may be subject to automated or manual analysis. Do not include personally identifiable information, confidential data, or proprietary materials in coursework submissions.

## 8. Miscellaneous Information

 - This is a 4 credit graduate level course offered by the Information and Decision Sciences department at UIC.
 - See the [academic calendar](http://catalog.uic.edu/ucat/academic-calendar/) for the semester timeline.
 - Students who wish to observe their religious holidays (http://oae.uic.edu/religious-calendar/) should notify the instructor within one week of the first lecture date. 
 - Contact the instructor at the earliest, if you require any accommodations for access to and/or participation in this course (see additional details below).
 - Refer to the academic integrity guidelines set by the university.
 
### 8.1 Special Accommodations

UIC has the [Disability Resource Center (DRC)](https://drc.uic.edu/) to help students needing special accommodations. If you have any condition, such as a physical or learning disability, which will make it difficult for you to carry out the work as outlined in the syllabus or which will require academic accommodations, you can go to the DRC for assistance, and should do so at the beginning of the semester or as early as possible. 

If a student needs remote learning accommodation due to a physical condition, such as if they are immuno-compromised and especially vulnerable to Covid, then the student can submit a formal request for remote learning accommodation through the DRC. This is needed in order to maintain fairness and consistency in our attendance policy.