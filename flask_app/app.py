from flask import Flask, render_template, request, redirect, url_for
import mlflow
import pickle
import os
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv

# ─────────────────── Load DagsHub credentials from .env ────────────────────
load_dotenv()
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

if not DAGSHUB_USERNAME or not DAGSHUB_TOKEN:
    raise EnvironmentError("DAGSHUB_USERNAME or DAGSHUB_TOKEN not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
mlflow.set_tracking_uri("https://dagshub.com/aftabalam1210/mini-mlops-project.mlflow")

# ─────────────────── Text Pre-processing ───────────────────────────────────
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    return lemmatization(text)

# ─────────────────── Load Model ────────────────────────────────────────────
def get_latest_model_version(model_name):
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_name = "my_model"
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

# Load vectorizer
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# ─────────────────── Flask App ──────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def home():
    result = request.args.get("result")  # Get result from query string
    return render_template("index.html", result=result)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    cleaned_text = normalize_text(text)
    features = vectorizer.transform([cleaned_text])
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])
    result = model.predict(features_df)
    
    # Redirect to home with result
    return redirect(url_for("home", result=result[0]))

# ─────────────────── Run App ───────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
