# Multimodal Depression Detection using Fine-Tuned RoBERTa and Wav2Vec2

## Overview

This project presents a transformer-based multimodal deep learning framework for automated depression detection using both textual and acoustic information extracted from clinical interviews.

The system combines:

- RoBERTa for contextual text understanding
- Wav2Vec2 for speech representation learning
- Multimodal feature fusion for joint reasoning
- Stratified cross-validation for robust evaluation

The project is built using the DAIC-WOZ dataset and focuses on improving depression detection performance through controlled fine-tuning, regularization, and multimodal representation learning.

This work was developed as part of undergraduate research in Artificial Intelligence, Natural Language Processing, and Mental Health Informatics.

---

# Research Motivation

Traditional depression diagnosis methods depend heavily on subjective clinical assessments and self-reported questionnaires. These approaches are often:

- Time-consuming
- Difficult to scale
- Dependent on human interpretation
- Inconsistent across evaluators

Recent advances in AI and NLP enable automated mental health analysis using language and speech patterns. However, many existing systems rely on a single modality and fail to capture the complementary relationship between what a person says and how they say it.

This project addresses that limitation through multimodal transformer-based learning.

---

# Problem Statement

The objective of this research is to design a robust multimodal AI system capable of detecting depression using both textual and acoustic features extracted from interview data.

The proposed system aims to:

- Improve prediction accuracy
- Reduce overfitting
- Handle class imbalance
- Enhance generalization capability
- Support scalable and objective mental health assessment

---

# Key Features

- Multimodal transformer architecture
- Fine-tuned RoBERTa text encoder
- Fine-tuned Wav2Vec2 audio encoder
- Text + speech feature fusion
- 3-fold stratified cross-validation
- Early stopping and learning rate scheduling
- Class imbalance handling using weighted BCE loss
- Gradient clipping and dropout regularization
- ROC-AUC, F1-score, confusion matrix evaluation
- Robust preprocessing pipeline for transcripts and audio

---

# Dataset

## DAIC-WOZ Dataset

The project uses the DAIC-WOZ (Distress Analysis Interview Corpus - Wizard of Oz) dataset.

The dataset contains:

- Clinical interview transcripts
- Audio recordings
- PHQ-8 depression labels

The dataset is widely used in depression detection research and serves as a benchmark dataset for multimodal mental health analysis.

Dataset Link:
https://dcapswoz.ict.usc.edu/

---

# System Architecture

The proposed architecture contains two parallel processing branches:

## 1. Text Processing Branch
- Input: Interview transcripts
- Model: RoBERTa-base
- Output: Contextual text embeddings

## 2. Audio Processing Branch
- Input: Raw audio waveform
- Model: Wav2Vec2-base
- Output: Acoustic feature embeddings

## 3. Feature Projection
- Dimensionality reduction:
  - 768 → 256
- Layer normalization
- ReLU activation

## 4. Multimodal Fusion
- Element-wise addition fusion
- Shared embedding representation

## 5. Classification Head
- Fully connected neural network
- ReLU activation
- Dropout regularization
- Sigmoid output layer

---

# Model Architecture

```text
Transcript → RoBERTa → Text Embeddings ┐
                                        ├→ Fusion → Classifier → Depression Prediction
Audio → Wav2Vec2 → Audio Embeddings ┘
```

---

# Training Strategy

## Fine-Tuning Strategy

To reduce overfitting caused by limited dataset size:

- Most transformer layers remain frozen
- Only the final encoder layers are fine-tuned

## Optimization

- Optimizer: AdamW
- Scheduler: ReduceLROnPlateau
- Loss Function: BCEWithLogitsLoss
- Positive Class Weighting
- Gradient Clipping
- Early Stopping

## Validation Strategy

- 3-Fold Stratified Cross Validation
- Maintains class distribution across folds
- Improves robustness and generalization

---

# Performance Metrics

The system is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Precision-Recall Curves

---

# Experimental Results

## Final Results

| Metric | Score |
|---|---|
| Accuracy | 88.8% |
| F1-score | 83.6% |
| ROC-AUC | 91.6% |

### Observations

- Multimodal learning outperformed single-modality approaches
- Audio features significantly improved recall
- Early stopping reduced overfitting
- Controlled fine-tuning improved stability

---

# Technologies Used

## Programming Language
- Python 3.8+

## Frameworks & Libraries
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- NumPy
- Pandas
- Librosa
- SoundFile
- Matplotlib

## Models
- RoBERTa-base
- Wav2Vec2-base

## Development Environment
- Google Colab
- Jupyter Notebook
- VS Code

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/multimodal-depression-detection.git
cd multimodal-depression-detection
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Execute Training

```bash
python pbl4.py
```

The pipeline will:

- Load transcripts and audio
- Preprocess data
- Train multimodal model
- Perform cross-validation
- Generate evaluation metrics
- Plot performance curves

---

# Important Research Contributions

This work contributes to depression detection research through:

- Transformer-based multimodal fusion
- Controlled fine-tuning for small datasets
- Robust evaluation using stratified validation
- Improved handling of class imbalance
- Integration of semantic and acoustic reasoning
- Scalable AI-driven mental health analysis

---

# Limitations

- Limited dataset size
- Clinical validation not performed
- Restricted to DAIC-WOZ dataset
- High computational requirements for transformers

---

# Future Enhancements

Potential future improvements include:

- Real-time depression monitoring
- Facial expression integration
- Larger clinical datasets
- Temporal conversational modeling
- Explainable AI integration
- Attention visualization
- Deployment as healthcare support tool
- Multilingual support

---

# Ethical Considerations

This project is intended strictly for research and educational purposes.

The system is designed to support mental health analysis and should not be used as a replacement for professional clinical diagnosis.

All dataset usage follows academic and research guidelines.

---

# Research Status

Paper submitted to conference review.

Code and implementation are released for research and educational purposes.

---

# Citation

If you use this work in your research, please cite:

```bibtex
@article{firoz2026multimodal,
  title={Multimodal Depression Detection using Fine-Tuned RoBERTa and Wav2Vec2},
  author={Firoz, Laraib},
  year={2026}
}
```

---

# Author

## Laraib Firoz
B.Tech Computer and Communication Engineering  
Manipal University Jaipur

Research Interests:
- Artificial Intelligence
- NLP
- Multimodal Learning
- Mental Health AI
- Transformer Architectures
- Deep Learning

---

# License

This project is released under the MIT License.
