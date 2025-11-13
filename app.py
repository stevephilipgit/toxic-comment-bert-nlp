import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import re, string, html
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# 1️⃣ Load NLTK Components
# -----------------------------
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# 2️⃣ Preprocessing Function
# -----------------------------
URL_RE = re.compile(r"http\S+|www.\S+")
PUNCT_TABLE = str.maketrans('', '', string.punctuation)

CONTRACTIONS = {
    "i'm": "i am", "you're": "you are", "can't": "cannot", "won't": "will not",
    "it's": "it is", "that's": "that is", "don't": "do not"
}

def expand_contractions(text):
    for c, e in CONTRACTIONS.items():
        text = text.replace(c, e)
    return text

def preprocess_text(text):
    text = text.lower()
    text = html.unescape(text)
    text = URL_RE.sub("", text)
    text = expand_contractions(text)
    text = text.translate(PUNCT_TABLE)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    tokens = [lemmatizer.lemmatize(t) for t in text.split() if t not in STOPWORDS]
    return " ".join(tokens).strip()

# -----------------------------
# 3️⃣ Load Model from Folder
# -----------------------------
MODEL_PATH = "stevehugss/toxic-comment-bert"   # change if needed

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)

    # Load model.safetensors
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_PATH,
        from_safetensors=True
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# 4️⃣ Prediction Function
# -----------------------------
def predict(text):
    clean_text = preprocess_text(text)

    encoded = tokenizer(
        clean_text,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=1)

    pred = probs.argmax().item()
    confidence = probs[0][pred].item()

    label = "Toxic 😡" if pred == 1 else "Non-Toxic 😇"

    return label, confidence, clean_text

# -----------------------------
# 5️⃣ Streamlit UI
# -----------------------------
st.set_page_config(page_title="Toxic Comment Detector", layout="centered")

st.title("🛡️ Toxic Comment Detection (Fine-tuned DistilBERT)")
st.write("Paste any comment below. AI will classify it as **Toxic** or **Non-Toxic**.")

text_input = st.text_area("Enter a comment:", height=150)

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        label, conf, clean_text = predict(text_input)

        st.subheader("🔮 Prediction")
        st.write(f"**Result:** {label}")
        st.write(f"**Confidence:** {conf:.2f}")

        st.subheader("🧹 Preprocessed Text")
        st.code(clean_text)

st.markdown("---")
st.write("Model fine-tuned on Civil Comments Dataset.")
