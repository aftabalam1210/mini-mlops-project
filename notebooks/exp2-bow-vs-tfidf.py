# Import necessary libraries
import mlflow
import mlflow.sklearn
from mlflow.data import from_pandas

import dagshub
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# DagsHub + MLflow connection
mlflow.set_tracking_uri('https://dagshub.com/aftabalam1210/mini-mlops-project.mlflow')
dagshub.init(repo_owner='aftabalam1210', repo_name='mini-mlops-project', mlflow=True)

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv').drop(columns=['tweet_id'])

# ---------------------------------------
# Text Preprocessing Functions
# ---------------------------------------

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in str(text).split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        return df
    except Exception as e:
        print(f'Error during text normalization: {e}')
        raise

# ---------------------------------------
# Preprocessing Pipeline
# ---------------------------------------
df = normalize_text(df)
df = df[df['sentiment'].isin(['happiness', 'sadness'])]
df['sentiment'] = df['sentiment'].replace({'sadness': 0, 'happiness': 1})

# ---------------------------------------
# MLflow Experiment Setup
# ---------------------------------------
mlflow.set_experiment("Bow vs TfIdf")

# Feature extraction approaches
vectorizers = {
    'BoW': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

# ML Algorithms
algorithms = {
    'LogisticRegression': LogisticRegression(),
    'MultinomialNB': MultinomialNB(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}

# ---------------------------------------
# Parent MLflow run
# ---------------------------------------
with mlflow.start_run(run_name="All Experiments") as parent_run:
    for algo_name, algorithm in algorithms.items():
        for vec_name, vectorizer in vectorizers.items():
            with mlflow.start_run(run_name=f"{algo_name} with {vec_name}", nested=True) as child_run:
                
                # Extract features
                X = vectorizer.fit_transform(df['content'])
                y = df['sentiment']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Log input dataset - ✅ NEW
                mlflow.log_input(from_pandas(df, source="processed_tweet_emotions.csv", name="preprocessed_dataset"))

                # Log parameters
                mlflow.log_param("vectorizer", vec_name)
                mlflow.log_param("algorithm", algo_name)
                mlflow.log_param("test_size", 0.2)
                
                # Train model
                model = algorithm
                model.fit(X_train, y_train)
                
                # Log model-specific parameters
                if algo_name == 'LogisticRegression':
                    mlflow.log_param("C", model.C)
                elif algo_name == 'MultinomialNB':
                    mlflow.log_param("alpha", model.alpha)
                elif algo_name == 'XGBoost':
                    mlflow.log_param("n_estimators", model.n_estimators)
                    mlflow.log_param("learning_rate", model.learning_rate)
                elif algo_name == 'RandomForest':
                    mlflow.log_param("n_estimators", model.n_estimators)
                    mlflow.log_param("max_depth", model.max_depth)
                elif algo_name == 'GradientBoosting':
                    mlflow.log_param("n_estimators", model.n_estimators)
                    mlflow.log_param("learning_rate", model.learning_rate)
                    mlflow.log_param("max_depth", model.max_depth)
                
                # Predict and evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                # Log model
                mlflow.sklearn.log_model(model, artifact_path="model")

                # Print summary
                print(f"\n📊 Algorithm: {algo_name} | Vectorizer: {vec_name}")
                print(f"✅ Accuracy: {accuracy:.4f}")
                print(f"✅ Precision: {precision:.4f}")
                print(f"✅ Recall: {recall:.4f}")
                print(f"✅ F1 Score: {f1:.4f}")
