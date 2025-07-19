# Import necessary libraries
import mlflow
import mlflow.sklearn
from mlflow.data import from_pandas

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pandas as pd
import re
import string
import dagshub
import os

# DagsHub tracking URI
mlflow.set_tracking_uri('https://dagshub.com/aftabalam1210/mini-mlops-project.mlflow')
dagshub.init(repo_owner='aftabalam1210', repo_name='mini-mlops-project', mlflow=True)

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv').drop(columns=['tweet_id'])

# -------------------------------
# Preprocessing Functions
# -------------------------------
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word.lower() not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return text.lower()

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    return re.sub('\s+', ' ', text).strip()

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalize_text(df):
    df = df.copy()
    df['content'] = df['content'].apply(lower_case)
    df['content'] = df['content'].apply(remove_stop_words)
    df['content'] = df['content'].apply(removing_numbers)
    df['content'] = df['content'].apply(removing_punctuations)
    df['content'] = df['content'].apply(removing_urls)
    df['content'] = df['content'].apply(lemmatization)
    return df

# -------------------------------
# Preprocess and filter dataset
# -------------------------------
df = normalize_text(df)
df = df[df['sentiment'].isin(['happiness', 'sadness'])]
df['sentiment'] = df['sentiment'].map({'sadness': 0, 'happiness': 1})

# -------------------------------
# Vectorization and Split
# -------------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['content'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# MLflow Experiment Setup
# -------------------------------
mlflow.set_experiment("LoR Hyperparameter Tuning")

# Define hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']  # supports both l1 and l2
}

# -------------------------------
# Start MLflow parent run
# -------------------------------
with mlflow.start_run(run_name="Grid Search Logistic Regression") as parent_run:

    # ✅ Save and log input dataset
    input_file_path = "preprocessed_tweet_emotions.csv"
    df.to_csv(input_file_path, index=False)
    mlflow.log_artifact(input_file_path, artifact_path="input_data")
    mlflow.log_input(from_pandas(df, source=input_file_path, name="preprocessed_dataset"))

    # Perform GridSearchCV on training data (not test data)
    grid_search = GridSearchCV(
        estimator=LogisticRegression(),
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # 🚀 Log each parameter combination as child run
    for params, mean_score, std_score in zip(grid_search.cv_results_['params'],
                                              grid_search.cv_results_['mean_test_score'],
                                              grid_search.cv_results_['std_test_score']):
        with mlflow.start_run(run_name=f"LR with {params}", nested=True):
            model = LogisticRegression(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Log metrics and params
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_score", mean_score)
            mlflow.log_metric("std_cv_score", std_score)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            print(f"\n🔍 Params: {params}")
            print(f"CV F1: {mean_score:.4f} ± {std_score:.4f} | Test F1: {f1:.4f}")

    # ✨ Log best model from GridSearch
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    mlflow.log_params(best_params)
    mlflow.log_metric("best_f1_score", best_score)

    mlflow.sklearn.log_model(best_model, artifact_path="model")

    print("\n✅ Best Hyperparameters:", best_params)
    print(f"🏆 Best CV F1 Score: {best_score:.4f}")
