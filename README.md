# 🛡️ Toxic Comment Detection using Fine-Tuned DistilBERT  
### Predict Toxic / Non-Toxic comments using NLP + Deep Learning

This project builds an **end-to-end toxic comment detection system** using:

- ⚡ **Baseline Logistic Regression (TF-IDF)**
- 🤖 **Fine-Tuned DistilBERT Transformer**
- 🧹 Heavy text preprocessing
- 📊 Performance evaluation & comparison
- 🌐 Deployed **Streamlit Web App**
- ☁️ Model hosted on **Hugging Face Hub**

The final model is a **fine-tuned DistilBERT** capable of identifying toxic comments such as:
- Hate speech  
- Harassment  
- Profanity  
- Abusive language  
- Threats  
- Insults  

---

# 🚀 Live Demo (Streamlit App)

🔗 **Live App:** *[Add your streamlit URL here]*  
Paste any comment and instantly see toxicity + confidence!

---

# 🤗 Hugging Face Model  
Your fine-tuned model is publicly available:

🔗 **https://huggingface.co/stevehugss/toxic-comment-bert**

This allows anyone to:
- Download the model  
- Use it for inference  
- Integrate into applications  

---

# 🎯 Project Aim

The goal was to build a **robust, real-world toxicity detection system** that:

✔ Understands context  
✔ Detects subtle insults  
✔ Handles sarcasm  
✔ Performs well on social media-style text  
✔ Provides easy-to-use inference via a web UI  

Baseline Machine Learning models are fast but fail on contextual toxicity.  
BERT-based transformers capture deep semantic meaning → much higher accuracy.

---

# 📦 Dataset  
Dataset used: **Civil Comments Toxicity Dataset (HuggingFace: `civil_comments`)**

- ~900K real comments  
- Contains a continuous toxicity score (0–1)  
- Converted to binary:  
  - **1 = Toxic** (score > 0.5)  
  - **0 = Non-Toxic**  
- Balanced sampled dataset for training  
  - 10,000 toxic  
  - 10,000 non-toxic  

---

# 🧹 Text Preprocessing Pipeline

The text cleaning used for Logistic Regression & BERT fine-tuning:

- Lowercasing  
- Remove URLs  
- Remove punctuation  
- Expand contractions (e.g., “can't” → “cannot”)  
- Remove numbers  
- Reduce repeated characters (“nooooo” → “noo”)  
- Tokenization  
- Stopword removal  
- Lemmatization  
- Merge cleaned tokens back into `clean_text`

This ensures the model receives clean, normalized text.

---

# 🧠 Model Architecture

## 1️⃣ **Baseline: Logistic Regression**
- Vectorizer: **TF-IDF (1-gram & 2-gram)**
- Resampling: RandomOversampler
- Limitations:
  - Cannot understand context  
  - Fails on subtle insults  
  - No semantic understanding  

---

## 2️⃣ **Fine-Tuned DistilBERT**
- Base model: `distilbert-base-uncased`
- Added dropout (0.3)
- Gradient checkpointing (memory optimized)
- Mixed-precision training (FP16)
- Max length: 64 tokens (optimized)
- Optimizer: AdamW
- Scheduler: Cosine learning rate decay
- Early Stopping: patience=2
- 20,000 samples (balanced)

---

# 📊 Results

## ⭐ Logistic Regression (Baseline)
