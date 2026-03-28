import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MOVIES_CSV = os.path.join(DATA_DIR, "movies.csv")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "cinevi_model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")


GENRE_TO_MOOD: Dict[str, str] = {
    "Comedy": "happy",
    "Romance": "romantic",
    "Drama": "sad",
    "Action": "energetic",
    "Thriller": "tense",
    "Horror": "dark",
    "Adventure": "excited",
}

FALLBACK_MOOD = "calm"


def load_movies(csv_path: str = MOVIES_CSV) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Movie dataset not found at '{csv_path}'. "
            "Please place the MovieLens small 'movies.csv' file in the data/ directory."
        )

    df = pd.read_csv(csv_path)

    # Basic cleaning: keep only required columns, drop nulls
    expected_cols = {"title", "genres"}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(sorted(missing_cols))}"
        )

    df = df[["title", "genres"]].copy()
    df = df.dropna(subset=["title", "genres"])

    # Normalize genre separators to a consistent form (pipe-separated)
    df["genres"] = df["genres"].astype(str).str.replace(",", "|")

    # Remove any placeholder genres often present in MovieLens
    df = df[df["genres"].str.lower() != "(no genres listed)"]

    return df.reset_index(drop=True)


def map_genres_to_mood(genres: str) -> str:
    tokens: List[str] = [g.strip() for g in str(genres).split("|") if g.strip()]
    for token in tokens:
        if token in GENRE_TO_MOOD:
            return GENRE_TO_MOOD[token]
    return FALLBACK_MOOD


def prepare_training_data(df: pd.DataFrame):
    df = df.copy()
    df["mood"] = df["genres"].apply(map_genres_to_mood)

    X_raw = df["genres"].tolist()
    y_raw = df["mood"].tolist()

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X_raw)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    return df, X, y, vectorizer, label_encoder


def train_model(X, y) -> LogisticRegression:
    clf = LogisticRegression(max_iter=200, multi_class="auto")
    clf.fit(X, y)
    return clf


def save_artifacts(
    model: LogisticRegression,
    vectorizer: CountVectorizer,
    label_encoder: LabelEncoder,
) -> None:
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)


def main() -> None:
    print("Loading movie dataset...")
    movies_df = load_movies()

    print("Preparing training data...")
    movies_df, X, y, vectorizer, label_encoder = prepare_training_data(movies_df)

    print("Training Logistic Regression model...")
    model = train_model(X, y)

    print("Saving model artifacts...")
    save_artifacts(model, vectorizer, label_encoder)

    print("Training complete.")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved vectorizer to: {VECTORIZER_PATH}")
    print(f"Saved label encoder to: {LABEL_ENCODER_PATH}")

    # Save a cleaned copy of the dataset with mood labels for the GUI to use
    output_csv = os.path.join(DATA_DIR, "movies_with_moods.csv")
    movies_df.to_csv(output_csv, index=False)
    print(f"Saved labeled movie data to: {output_csv}")


if __name__ == "__main__":
    main()

