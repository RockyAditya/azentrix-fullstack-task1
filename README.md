# 🎬 CineScore: AI-Powered Sentiment Analysis of IMDB Movie Reviews using DistilBERT

## 📌 Project Overview

CineScore is an end-to-end Natural Language Processing (NLP) project that performs sentiment analysis on movie reviews from the IMDB dataset. The project compares a traditional Machine Learning approach (TF-IDF + SGDClassifier) with a Transformer-based Deep Learning approach (DistilBERT) to classify reviews as Positive or Negative.

The objective is to evaluate the effectiveness of modern transformer architectures against conventional machine learning techniques and deploy the best-performing model through an interactive Streamlit web application.

---

## 🎯 Objectives

- Perform sentiment classification on movie reviews.
- Compare traditional Machine Learning and Transformer-based NLP approaches.
- Fine-tune DistilBERT on the IMDB dataset.
- Evaluate models using standard classification metrics.
- Deploy the trained model using Streamlit.
- Build a professional AI-powered dashboard for sentiment prediction.

---

## 📂 Dataset

**Dataset:** IMDB Movie Review Dataset

- Total Reviews: 50,000
- Positive Reviews: 25,000
- Negative Reviews: 25,000
- Training Samples: 40,000
- Validation Samples: 10,000

Dataset Source:
https://ai.stanford.edu/~amaas/data/sentiment/

---

## 🛠 Technologies Used

### Programming Language
- Python

### Libraries & Frameworks
- Pandas
- NumPy
- Scikit-Learn
- PyTorch
- Hugging Face Transformers
- Datasets
- Evaluate
- Streamlit
- Plotly
- Matplotlib
- Seaborn

### Model
- DistilBERT (`distilbert-base-uncased`)

### Development Environment
- Google Colab
- VS Code

---

# 🔄 Project Workflow

## 1. Data Collection

- Loaded IMDB Movie Review Dataset.
- Verified class balance.
- Performed train-validation split.

## 2. Data Preprocessing

- Removed HTML tags.
- Converted text to lowercase.
- Cleaned review content.
- Encoded labels:
  - Positive → 1
  - Negative → 0

## 3. Baseline Model

### TF-IDF + SGDClassifier

Traditional Machine Learning pipeline:

Review Text
→ TF-IDF Vectorization
→ SGDClassifier
→ Sentiment Prediction

### Baseline Result

| Metric | Score |
|----------|----------|
| Accuracy | 88.79% |

---

## 4. Transformer Model

### DistilBERT

DistilBERT is a lightweight version of BERT created through knowledge distillation.

Advantages:

- Faster training
- Reduced memory consumption
- Strong contextual understanding
- High NLP performance

Architecture:

Movie Review
↓
Tokenizer
↓
DistilBERT Encoder
↓
Classification Head
↓
Positive / Negative

---

## ⚙ Training Configuration

| Parameter | Value |
|------------|---------|
| Model | distilbert-base-uncased |
| Epochs | 3 |
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| Optimizer | AdamW |
| GPU | Tesla T4 |
| Framework | PyTorch + Hugging Face |

---

# 📊 Results

## DistilBERT Performance

| Metric | Score |
|----------|----------|
| Accuracy | 92.16% |
| Precision | 91.83% |
| Recall | 92.56% |
| F1 Score | 92.19% |

---

## Confusion Matrix

| Actual / Predicted | Negative | Positive |
|-------------------|----------|----------|
| Negative | 4588 | 412 |
| Positive | 372 | 4628 |

### Summary

- Correct Predictions: 9216
- Incorrect Predictions: 784
- Validation Samples: 10,000

---

# 📈 Model Comparison

| Model | Accuracy |
|---------|---------|
| TF-IDF + SGDClassifier | 88.79% |
| DistilBERT | 92.16% |

### Improvement

DistilBERT achieved:

**+3.37% Accuracy Improvement**

over the traditional TF-IDF + SGDClassifier baseline.

---

# 🔍 Sample Predictions

### Example 1

Input:

```text
I absolutely loved this movie. The acting was brilliant.
```

Prediction:

```text
Positive
Confidence: 99.82%
```

### Example 2

Input:

```text
This was the worst movie I have ever watched.
```

Prediction:

```text
Negative
Confidence: 99.79%
```

---

# 🎨 Streamlit Application

The project includes a professional Streamlit dashboard named **CineScore**.

### Features

- Interactive Sentiment Prediction
- Confidence Visualization
- Performance Dashboard
- DistilBERT Architecture Overview
- TF-IDF vs DistilBERT Comparison
- Project Analytics
- Modern Dark-Themed UI

---

# 📁 Project Structure

```text
CineScore/
│
├── app.py
├── requirements.txt
├── README.md
│
├── final_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── vocab.txt
│   └── training_args.bin
│
├── graphs/
│   ├── accuracy_curve.png
│   ├── confusion_matrix.png
│   ├── performance_metrics.png
│   ├── prediction_distribution.png
│   ├── training_loss.png
│   └── validation_loss.png
│
└── notebooks/
```

---

# 🚀 Running the Application

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

Windows:

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Streamlit

```bash
streamlit run app.py
```

---

# 💡 Key Learnings

- NLP preprocessing techniques
- Feature extraction using TF-IDF
- Transformer-based sentiment classification
- Fine-tuning DistilBERT
- Model evaluation and comparison
- Streamlit deployment
- Interactive AI dashboard development

---

# 🔮 Future Enhancements

- Multi-class sentiment analysis
- Aspect-based sentiment analysis
- Real-time review monitoring
- Hugging Face Spaces deployment
- Streamlit Cloud deployment
- Review recommendation system

---

# 👨‍💻 Author

**Aditya**

B.Tech Computer Science & Engineering

Specialization:
- Artificial Intelligence
- Machine Learning
- Natural Language Processing
- Full Stack Development

---

# 🏆 Final Outcome

✅ Traditional ML Baseline Developed

✅ DistilBERT Fine-Tuned on 50K Reviews

✅ Achieved 92.16% Accuracy

✅ Model Comparison Completed

✅ Streamlit Dashboard Developed

✅ End-to-End NLP Pipeline Implemented

**Project Status: Completed**# azentrix-fullstack-task1
