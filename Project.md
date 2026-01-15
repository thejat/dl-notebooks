# Project Instructions

This document provides comprehensive guidance for your course project. Please read all sections carefully before starting.

## Table of Contents

1. [Aim](#aim)
2. [Choosing Your Project Topic](#choosing-your-project-topic)
3. [Dataset Guidelines](#dataset-guidelines)
4. [Project Outcomes & Deliverables](#project-outcomes)
5. [Report Templates](#report-templates)
6. [Suggested Timeline](#suggested-timeline)
7. [Technical Expectations](#technical-expectations)
8. [Common Mistakes to Avoid](#common-mistakes-to-avoid)
9. [Team Collaboration Guidelines](#team-collaboration-guidelines)
10. [Presentation Guidelines](#presentation-guidelines)
11. [Grading Rubric](#grading-rubric)
12. [Pre-Submission Checklist](#pre-submission-checklist)
13. [Frequently Asked Questions](#frequently-asked-questions)

---

## Aim

The aim of the project is to simulate the real-world process of conceptualizing a data analytics project and bringing unique insights using deep learning. More specifically, the project component of this course allows you to:

- **Explore** a dataset of your choosing
- **Build** statistical/deep learning model(s) to achieve a meaningful goal
- **Report** your experience, findings, and insights
- **Present** your work professionally

### What Makes a Successful Project?

| Aspect | Description |
|--------|-------------|
| **Clear Problem** | Well-defined question with measurable outcomes |
| **Appropriate Data** | Sufficient quality and quantity for deep learning |
| **Sound Methodology** | Proper experimental design with baselines |
| **Insightful Analysis** | Going beyond just reporting numbers |
| **Clear Communication** | Well-written report and engaging presentation |

---

## Choosing Your Project Topic

### Good Project Characteristics ✅

- Clear problem definition with measurable outcomes
- Appropriate dataset availability (sufficient size and quality)
- Alignment with deep learning techniques covered in class
- Feasibility within the semester timeline
- Opportunity for creative contributions

### Example Project Categories

| Category | Example Topics | Typical Architectures |
|----------|---------------|----------------------|
| **Computer Vision** | Medical image classification, Object detection, Art style transfer, Image segmentation | CNNs, ResNet, VGG, U-Net |
| **Natural Language Processing** | Sentiment analysis, Document summarization, Question answering, Named entity recognition | RNNs, LSTMs, Transformers, BERT |
| **Time Series** | Stock prediction, Weather forecasting, Anomaly detection, Energy consumption prediction | LSTMs, GRUs, Temporal CNNs |
| **Multimodal** | Image captioning, Visual question answering, Video understanding | CNN+RNN combinations, Transformers |
| **Generative Models** | Image generation, Text generation, Data augmentation | VAEs, GANs, Diffusion models |
| **Reinforcement Learning** | Game playing, Robot control, Recommendation systems | DQN, Policy Gradient methods |

### Topics to Avoid ❌

- Projects requiring proprietary or unavailable data
- Overly ambitious scope (e.g., "Build a complete autonomous driving system")
- Projects that are essentially running existing tutorials without modification
- Topics with no clear evaluation criteria
- Projects requiring specialized hardware you don't have access to

---

## Dataset Guidelines

### Minimum Requirements

| Aspect | Guideline |
|--------|----------|
| **Size** | Generally 1,000+ samples for classification; more for complex tasks |
| **Quality** | Clean, well-documented, appropriate for your problem |
| **Splits** | Plan for train/validation/test splits (e.g., 70/15/15 or 80/10/10) |
| **Labels** | Verified and consistent labeling for supervised tasks |

### Recommended Dataset Sources

- [Kaggle Datasets](https://kaggle.com/datasets) - Wide variety of datasets with community discussions
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml) - Classic ML datasets
- [Hugging Face Datasets](https://huggingface.co/datasets) - Excellent for NLP tasks
- [Google Dataset Search](https://datasetsearch.research.google.com) - Search engine for datasets
- [Papers with Code Datasets](https://paperswithcode.com/datasets) - Datasets linked to benchmarks
- [TensorFlow Datasets](https://www.tensorflow.org/datasets) - Ready-to-use datasets
- [AWS Open Data](https://registry.opendata.aws/) - Large-scale datasets

### Data Ethics Considerations

- ✅ Ensure you have rights to use the data
- ✅ Consider privacy implications (especially for face/medical data)
- ✅ Document data provenance and licensing
- ✅ Be aware of potential biases in the data
- ❌ Never use data without proper authorization

---

## Project Outcomes

There are **two due-dates** for project deliverables: one intermediate and one final. See the course logistics page for the exact dates.

### Intermediate Deliverable

**Project Scope and Plan** (Maximum 2 pages, 12 point font, single column)

Your intermediate deliverable should include:

1. **Problem Statement**: Clear description of your project idea (more than one related idea is acceptable)
2. **Deep Learning Suitability**: Why is deep learning appropriate for this problem?
3. **Exploratory Data Analysis**:
   - Dataset source and description
   - Number of samples and features
   - Class distribution (for classification) or target distribution (for regression)
   - Sample visualizations
   - Missing data analysis
4. **Project Plan**: Detailed timeline with specific tasks, deadlines, and team member assignments

### Final Deliverables

#### 1. Project Report (Maximum 8 pages + optional appendix)

- 12 point font, single column
- Appendix for supplementary material (may or may not be checked)
- Explain your creative contributions (modeling, optimization, inference, analysis, insights)
- Include technical discussion of what worked, what didn't, and why
- Can optionally be combined with code as a Jupyter notebook

#### 2. Code and Data

- Well-documented Jupyter notebook(s) or Python scripts
- Small sample of the data (or clear instructions to obtain it)
- README with setup instructions
- Requirements file (requirements.txt or environment.yml)

#### 3. Presentation

- Delivered at the end of the semester (see course syllabus for date)
- Using Jupyter notebook or slides
- Cover the whole project and key learnings

---

## Report Templates

### Intermediate Report Structure (2 pages max)

```
1. PROBLEM STATEMENT (~1/4 page)
   - What problem are you solving?
   - Why is it important/interesting?
   - What is your hypothesis?

2. DATASET DESCRIPTION (~1/2 page)
   - Source and access method
   - Size (number of samples, features)
   - Class/target distribution
   - Sample visualizations (1-2 figures)
   - Any data quality issues identified

3. PROPOSED APPROACH (~1/2 page)
   - Why is deep learning appropriate?
   - What architectures will you explore?
   - What baselines will you compare against?
   - What metrics will you use?

4. PROJECT PLAN (~3/4 page)
   - Week-by-week task breakdown
   - Team member responsibilities
   - Risk assessment and mitigation strategies
   - Computational resources needed
```

### Final Report Structure (8 pages max)

```
1. INTRODUCTION (~1 page)
   - Problem motivation and context
   - Summary of contributions
   - Report organization

2. RELATED WORK (~1/2 page)
   - Prior approaches to similar problems
   - How your work differs or builds upon them
   - Key references

3. DATASET AND PREPROCESSING (~1 page)
   - Detailed data description
   - Preprocessing pipeline
   - Data augmentation strategies
   - Train/validation/test splits

4. METHODOLOGY (~2 pages)
   - Model architecture(s) with diagrams
   - Training procedure (optimizer, learning rate, epochs)
   - Hyperparameter choices and justification
   - Design decisions and rationale

5. EXPERIMENTS AND RESULTS (~2 pages)
   - Experimental setup
   - Evaluation metrics (with justification!)
   - Quantitative results (tables, learning curves)
   - Qualitative analysis (visualizations, examples)
   - Comparison with baselines

6. DISCUSSION (~1 page)
   - What worked well and why
   - What didn't work and why
   - Ablation studies
   - Limitations of your approach
   - Lessons learned

7. CONCLUSION (~1/2 page)
   - Summary of key findings
   - Potential future work
   - Broader implications

REFERENCES
   - Properly formatted citations
```

---

## Suggested Timeline

### Weekly Milestones

| Week | Phase | Key Activities | Checkpoint Question |
|------|-------|---------------|--------------------|
| 1-2 | **Planning** | Team formation, topic brainstorming, initial data exploration | Do we have a clear, feasible problem statement? |
| 3-4 | **Data Prep** | Data collection, cleaning, EDA, preprocessing pipeline | Is our data ready for modeling? |
| 5 | **Intermediate** | Write and submit intermediate report | Have we addressed all required sections? |
| 6-7 | **Baseline** | Implement baseline models, establish benchmarks | Does our baseline model work correctly? |
| 8-9 | **Development** | Model iterations, hyperparameter tuning, experimentation | Have we tried at least 3 model variations? |
| 10-11 | **Analysis** | Result analysis, visualization, ablation studies | What insights have we gained? |
| 12-13 | **Finalization** | Report writing, code cleanup, documentation | Is our report complete and well-written? |
| 14 | **Presentation** | Prepare and deliver presentation | Are we ready to present confidently? |

### Time Management Tips

- **Start early**: Data issues always take longer than expected
- **Set internal deadlines**: Finish 2-3 days before actual deadlines
- **Parallelize**: Team members can work on different experiments simultaneously
- **Document as you go**: Don't leave all writing for the end
- **Regular check-ins**: Weekly team meetings keep everyone aligned

---

## Technical Expectations

### Model Development Principles

1. **Baseline First**: Always implement a simple baseline before complex models
   - For classification: Logistic regression, simple CNN
   - For NLP: Bag-of-words + simple classifier
   - For time series: ARIMA, simple LSTM

2. **Iterative Improvement**: Document at least 3 model iterations with clear rationale

3. **Transfer Learning**: Do not train deep networks from scratch if it can be avoided
   - Use pre-trained models (ImageNet, BERT, etc.)
   - Fine-tune on your specific task

4. **Proper Evaluation**: Use appropriate metrics beyond just accuracy!

| Task Type | Recommended Metrics |
|-----------|--------------------|
| Classification | Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix |
| Regression | MSE, MAE, RMSE, R², residual plots |
| Generation | BLEU, ROUGE, FID, perplexity, human evaluation |
| Ranking | MAP, NDCG, MRR |

### Experiment Logging

Document all experiments systematically:

```python
# Example experiment configuration
experiment_config = {
    "experiment_name": "resnet18_finetuned_v2",
    "model": "ResNet18-pretrained",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
    "optimizer": "Adam",
    "weight_decay": 1e-4,
    "data_augmentation": ["RandomHorizontalFlip", "RandomRotation(10)"],
    "seed": 42,  # Always set random seeds for reproducibility!
    "notes": "Added dropout layer after conv3"
}
```

### Reproducibility Requirements

- ✅ Set random seeds for all sources of randomness
- ✅ Document all hyperparameters
- ✅ Version your code (use Git)
- ✅ Save model checkpoints
- ✅ Include requirements.txt or environment.yml
- ✅ Test that your code runs from scratch

---

## Common Mistakes to Avoid

### Data Issues ⚠️

| Mistake | Why It's Bad | How to Avoid |
|---------|-------------|-------------|
| Training on test data (data leakage) | Inflated metrics, won't generalize | Strict data splits before any processing |
| Not shuffling before splitting | Biased splits | Random shuffle with fixed seed |
| Ignoring class imbalance | Model biased toward majority class | Use stratified splits, weighted loss, oversampling |
| Fitting preprocessors on full data | Information leakage | Fit only on training data |
| Not checking for duplicates | Train/test overlap | Deduplicate before splitting |

### Modeling Issues ⚠️

| Mistake | Why It's Bad | How to Avoid |
|---------|-------------|-------------|
| No baseline model | Can't assess if deep learning helps | Always start with simple baselines |
| Overcomplicating too early | Wastes time, harder to debug | Start simple, add complexity gradually |
| Not monitoring overfitting | Poor generalization | Use validation set, early stopping |
| Ignoring computational constraints | Can't finish experiments | Plan compute budget, use cloud resources |
| Not saving checkpoints | Lost progress if crashes | Save model regularly |

### Reporting Issues ⚠️

| Mistake | Why It's Bad | How to Avoid |
|---------|-------------|-------------|
| Only reporting best results | Cherry-picking, not scientific | Report all experiments, include failures |
| Missing error bars/variance | Results may not be significant | Run multiple seeds, report std |
| Unclear visualizations | Can't interpret results | Label axes, add legends, use appropriate plots |
| No comparison to baselines | Can't assess contribution | Always include baseline comparisons |
| Missing citations | Plagiarism, incomplete context | Cite all sources meticulously |

### Project Management Issues ⚠️

| Mistake | Why It's Bad | How to Avoid |
|---------|-------------|-------------|
| Starting too late | Rushed, poor quality | Start immediately after topic approval |
| Poor team communication | Duplicated work, gaps | Regular meetings, shared docs |
| No version control | Lost code, merge conflicts | Use Git from day one |
| Single point of failure | Risk if one person unavailable | Cross-train, document everything |

---

## Team Collaboration Guidelines

### Version Control Best Practices

```bash
# Recommended Git workflow
git checkout -b feature/experiment-resnet50  # Create feature branch
# ... make changes ...
git add .
git commit -m "Add ResNet50 experiment with data augmentation"
git push origin feature/experiment-resnet50
# Create pull request for review
```

### Suggested Role Distribution

| Role | Responsibilities |
|------|------------------|
| **Data Lead** | Data collection, preprocessing, augmentation, data quality |
| **Model Lead** | Architecture design, training pipeline, hyperparameter tuning |
| **Evaluation Lead** | Metrics implementation, visualization, analysis, ablation studies |
| **Report Lead** | Writing, formatting, citations, presentation preparation |

*Note: Roles should overlap significantly. Every team member should understand all parts and contribute to coding!*

### Communication Tips

- 📅 Establish regular meeting times (at least weekly)
- 📝 Use shared documents for meeting notes and decisions
- 💬 Set up a team chat (Slack, Discord, etc.)
- 📊 Share progress via experiment tracking (Weights & Biases, TensorBoard)

### Parallelization Strategy

Team members can run multiple experiments simultaneously:
- Each member uses their own Google Colab session
- Coordinate to avoid duplicate experiments
- Share results in a common spreadsheet or tracking tool
- Example: One member tunes learning rate while another tests architectures

---

## Presentation Guidelines

### Format

- **Duration**: Check syllabus for time allocation (typically 10-15 minutes + Q&A)
- **Format**: Slides or live Jupyter notebook demo
- **All team members should speak**

### Required Components

1. **Problem Motivation** (1-2 slides)
   - Why should we care about this problem?
   - Hook the audience with a compelling example or statistic

2. **Data Overview** (1-2 slides)
   - Show actual examples from your data!
   - Key statistics and visualizations

3. **Approach Summary** (2-3 slides)
   - Key architectural choices and why
   - Visual diagrams of your model

4. **Results** (2-3 slides)
   - Quantitative metrics in tables/charts
   - Qualitative examples (show predictions!)
   - Comparison with baselines

5. **Demo** (optional but recommended)
   - Live demonstration if applicable
   - Video backup in case of technical issues

6. **Insights and Lessons** (1-2 slides)
   - What worked and why?
   - What would you do differently?
   - Key takeaways

### Tips for Effective Presentations

- ✅ Use visualizations over text
- ✅ Practice timing beforehand
- ✅ Prepare for common questions (see FAQ section)
- ✅ Have backup slides for detailed questions
- ❌ Don't read from slides
- ❌ Don't include too much text
- ❌ Don't skip the demo/examples

### Common Q&A Questions to Prepare For

- "Why did you choose this architecture?"
- "How does this compare to state-of-the-art?"
- "What would you do differently with more time?"
- "How would this scale to production?"
- "What's the computational cost?"
- "How did you handle [specific data challenge]?"

---

## Grading Rubric

### Intermediate Deliverable

Graded based on:
- ✓ Complete project plan sufficiently described
- ✓ Clear and well-scoped problem statement
- ✓ Quality of exploratory data analysis
- ✓ Realistic timeline with clear team assignments
- ✓ Identification of potential risks and mitigation strategies

### Final Deliverable

The final deliverable is evaluated on **four dimensions** (the 4 C's):

| Criterion | What We Look For | Weight |
|-----------|-----------------|--------|
| **Correctness** | Valid experimental setup, appropriate metrics, technical accuracy, sound assumptions | 25% |
| **Content** | Novel contributions, project depth, understanding of topics, interesting insights | 25% |
| **Creativity** | Non-obvious solutions, unique approaches, innovative design choices | 25% |
| **Clarity** | Professional writing, clear structure, proper citations, good visualizations | 25% |

#### Detailed Breakdown

**Correctness:**
- Are the evaluation metrics appropriate for the problem?
- Is the experimental setup valid (no data leakage, proper splits)?
- Are the technical claims accurate?
- Are assumptions clearly stated and reasonable?

**Content:**
- Why this data? Why this problem?
- What novel contributions are made?
- Are there interesting visualizations and conclusions?
- Is there thoughtful discussion of methodology?

**Creativity:**
- How non-obvious is the solution?
- Were interesting design choices explored?
- Is there innovation in the approach?

**Clarity:**
- Is the writing professional and well-structured?
- Are references properly cited?
- Is the presentation clear and engaging?
- Are figures and tables well-designed?

### Important Policies

- **Citation Requirement**: All external material/sources (code/idea/theory/insights) must be cited. Failure to cite is academic dishonesty.
- **Allowed Resources**: Pre-trained models, databases, web servers, frontend frameworks, visualization tools are encouraged.
- **Discouraged**: Proprietary software (Matlab, Mathematica, etc.)
- **Exclusivity**: This project cannot be used for any other course or requirement.

---

## Pre-Submission Checklist

### Intermediate Deliverable ✓

- [ ] Problem is clearly defined with specific goals
- [ ] Dataset is accessible and fully described
- [ ] Exploratory analysis includes meaningful visualizations
- [ ] Deep learning suitability is justified
- [ ] Project plan has specific dates and task assignments
- [ ] Each team member's responsibilities are clear
- [ ] Risks and mitigation strategies are identified
- [ ] Document is within the 2-page limit
- [ ] All figures are readable and properly labeled
- [ ] Spelling and grammar are checked

### Final Deliverable ✓

**Report:**
- [ ] All required sections are present and complete
- [ ] Methodology is clearly explained
- [ ] Results include comparison with baselines
- [ ] Discussion addresses what worked and what didn't
- [ ] All figures are high quality and properly labeled
- [ ] All sources are properly cited
- [ ] Report is within the 8-page limit

**Code:**
- [ ] Code runs without errors from scratch
- [ ] Code is well-commented and organized
- [ ] README with setup instructions is included
- [ ] requirements.txt or environment.yml is provided
- [ ] Random seeds are set for reproducibility

**Data:**
- [ ] Sample data is included or clear download instructions provided
- [ ] Data preprocessing pipeline is documented

**Team:**
- [ ] Team contributions are documented
- [ ] All team members have reviewed final submission

---

## Frequently Asked Questions

### About Project Scope

**Q: Can we use pre-trained models?**
> **A:** Yes! Transfer learning is strongly encouraged. Document what pre-trained model you use, why you chose it, and how you adapted it for your task.

**Q: How much code should be original?**
> **A:** Focus on adaptation and experimentation rather than writing everything from scratch. Using library functions and pre-built components is fine and expected. However, copying entire solutions without understanding or modification is not acceptable.

**Q: Can we change topics after the intermediate submission?**
> **A:** Consult with the instructor. Minor pivots are usually acceptable if well-justified. Major changes may require approval and could affect your intermediate grade.

**Q: What if our results are poor?**
> **A:** Negative results with good analysis are valuable! Focus on:
> - Understanding *why* the approach didn't work
> - What you learned from the experience
> - What you would try differently given more time
> - Proper documentation of your experimental process

### About Data

**Q: Can we create our own dataset?**
> **A:** Yes, but ensure you have enough samples and the labeling is consistent. Document your data collection process thoroughly.

**Q: What if our dataset is too large to submit?**
> **A:** Submit a representative sample and provide clear instructions (and scripts if needed) to obtain the full dataset.

**Q: Can we use datasets from Kaggle competitions?**
> **A:** Yes, but make sure your project goes beyond just running the competition baseline. Add your own analysis, experiments, and insights.

### About Teamwork

**Q: How should we split work for grading?**
> **A:** Include a contribution statement in your report. While tasks may be divided, all members should understand all parts of the project and be able to answer questions about any aspect.

**Q: What if a team member isn't contributing?**
> **A:** Address issues early through direct communication. If problems persist, speak with the instructor before the final deadline.

### About Technical Aspects

**Q: Do we need GPU access?**
> **A:** For most projects, Google Colab's free GPU tier is sufficient. Plan your experiments to work within these constraints. For larger needs, consider Colab Pro or cloud credits.

**Q: What deep learning framework should we use?**
> **A:** PyTorch and TensorFlow/Keras are both acceptable. Use what you're most comfortable with or what has better support for your specific task.

**Q: How many experiments should we run?**
> **A:** Quality over quantity. At minimum:
> - 1+ simple baseline(s)
> - 1 main deep learning approach
> - 2-3 variations or ablations
> Document all experiments, including unsuccessful ones.

---

## Summary: Keys to Success 🔑

1. **Start early** and iterate often
2. **Establish baselines** before complex models
3. **Document everything** as you go
4. **Communicate regularly** with your team
5. **Use transfer learning** when possible
6. **Focus on insights**, not just metrics
7. **Ask for help** when stuck
8. **Cite all sources** meticulously

Good luck with your projects! 🚀