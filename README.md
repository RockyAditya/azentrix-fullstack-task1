# 🎬 CineScore: AI-Powered Sentiment Analysis of IMDB Movie Reviews using DistilBERT

## 📌 Project Overview

CineScore is an end-to-end Natural Language Processing (NLP) project that performs sentiment analysis on movie reviews from the IMDB dataset. The project compares a traditional Machine Learning approach (TF-IDF + SGDClassifier) with a Transformer-based Deep Learning approach (DistilBERT) to classify reviews as Positive or Negative.

---

## 🎯 Objectives

* Perform sentiment classification on movie reviews.
* Compare traditional ML and Transformer-based approaches.
* Fine-tune DistilBERT on IMDB reviews.
* Evaluate models using industry-standard metrics.
* Deploy the model using Streamlit.
* Host the trained model for reproducible deployment.

---

## 📂 Dataset

**Dataset:** IMDB Movie Review Dataset

* Total Reviews: 50,000
* Positive Reviews: 25,000
* Negative Reviews: 25,000
* Training Samples: 40,000
* Validation Samples: 10,000

Dataset Source:
https://ai.stanford.edu/~amaas/data/sentiment/

---

## 🛠 Technologies Used

### Languages & Frameworks

* Python
* PyTorch
* Hugging Face Transformers
* Scikit-Learn
* Streamlit
* Plotly
* Pandas
* NumPy

### Development Tools

* Google Colab
* Visual Studio Code
* GitHub
* Hugging Face Hub

---

# 🔄 Project Workflow

## 1. Data Preprocessing

* HTML tag removal
* Text cleaning
* Lowercase conversion
* Label encoding

## 2. Baseline Model

### TF-IDF + SGDClassifier

Movie Review
→ TF-IDF Vectorization
→ SGDClassifier
→ Sentiment Prediction

**Accuracy:** 88.79%

---

## 3. Transformer Model

### DistilBERT

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

| Parameter          | Value                   |
| ------------------ | ----------------------- |
| Base Model         | distilbert-base-uncased |
| Epochs             | 3                       |
| Learning Rate      | 2e-5                    |
| Optimizer          | AdamW                   |
| GPU                | Tesla T4                |
| Training Samples   | 40,000                  |
| Validation Samples | 10,000                  |

---

# 📊 Model Performance

## DistilBERT Results

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 92.16% |
| Precision | 91.83% |
| Recall    | 92.56% |
| F1 Score  | 92.19% |

### Classification Report

| Class    | Precision | Recall | F1 Score |
| -------- | --------- | ------ | -------- |
| Negative | 0.93      | 0.92   | 0.92     |
| Positive | 0.92      | 0.93   | 0.92     |

### Confusion Matrix

| Actual / Predicted | Negative | Positive |
| ------------------ | -------- | -------- |
| Negative           | 4588     | 412      |
| Positive           | 372      | 4628     |

---

# 📈 Model Comparison

| Model                  | Accuracy |
| ---------------------- | -------- |
| TF-IDF + SGDClassifier | 88.79%   |
| DistilBERT             | 92.16%   |

### Improvement

**DistilBERT achieved a +3.37% accuracy improvement over the baseline model.**

---

# 🤗 Trained Model Repository

The DistilBERT model used in this project was **fine-tuned by the author** on the IMDB Movie Review Dataset.

Due to GitHub's file size limitation (100 MB per file), the trained model artifacts (~256 MB) are hosted separately on the author's Hugging Face account.

> **Important:** The model hosted on Hugging Face is not a third-party model. It is the final fine-tuned model trained during this project and uploaded by the author.

### Model Repository

Hugging Face Model:

https://huggingface.co/DemonKing112/imdb-distilbert-sentiment

### Training Summary

| Parameter          | Value              |
| ------------------ | ------------------ |
| Base Model         | DistilBERT         |
| Dataset            | IMDB Movie Reviews |
| Training Samples   | 40,000             |
| Validation Samples | 10,000             |
| Epochs             | 3                  |
| Accuracy           | 92.16%             |
| F1 Score           | 92.19%             |

### Why Hugging Face?

The trained model contains a `model.safetensors` file of approximately **256 MB**, which exceeds GitHub's standard file size limit.

To maintain a lightweight repository while ensuring full reproducibility, the model is hosted on Hugging Face Hub and automatically downloaded by the Streamlit application.

---

# 🎨 Streamlit Application

The project includes a professional Streamlit dashboard called **CineScore**.

### Features

* Interactive Sentiment Prediction
* Confidence Visualization
* Model Analytics Dashboard
* TF-IDF vs DistilBERT Comparison
* DistilBERT Architecture Overview
* Responsive UI Design

---

# 📉 Visualizations

The project includes:

* Accuracy Curve
* Training Loss Curve
* Validation Loss Curve
* Performance Metrics Chart
* Prediction Distribution
* Confusion Matrix Heatmap

All graphs are available inside the `graphs/` directory.

---

# 📁 Project Structure

```text
azentrix-fullstack-task1/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── graphs/
│   ├── accuracy_curve.png
│   ├── confusion_matrix.png
│   ├── performance_metrics.png
│   ├── prediction_distribution.png
│   ├── training_loss.png
│   └── validation_loss.png
│
└── Hugging Face Model
    └── DemonKing112/imdb-distilbert-sentiment
```

---

# 🚀 Installation & Usage

```bash
git clone https://github.com/RockyAditya/azentrix-fullstack-task1.git
cd azentrix-fullstack-task1
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

The trained model will be automatically downloaded from Hugging Face during the first run.

---

# 💡 Key Learnings

* NLP Preprocessing
* TF-IDF Feature Engineering
* Transformer Fine-Tuning
* Hugging Face Ecosystem
* Streamlit Deployment
* Model Hosting & Distribution
* GitHub Project Management

---

# 👨‍💻 Author

**Aditya**

B.Tech Computer Science & Engineering

Areas of Interest:

* Artificial Intelligence
* Machine Learning
* Natural Language Processing
* Full Stack Development

---

# 🏆 Project Outcome

✅ TF-IDF Baseline Developed

✅ DistilBERT Fine-Tuned on 50K Reviews

✅ Achieved 92.16% Accuracy

✅ Comparative Analysis Completed

✅ Professional Streamlit Dashboard Built

✅ Trained Model Hosted on Hugging Face

✅ Source Code Published on GitHub

✅ Deployment-Ready Application

**Project Status: Completed**
