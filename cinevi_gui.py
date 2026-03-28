import os
import pickle
import random
import sys
from typing import Dict, List, Tuple

import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk


BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
LABELED_MOVIES_CSV = os.path.join(DATA_DIR, "movies_with_moods.csv")

MODEL_PATH = os.path.join(BASE_DIR, "cinevi_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")


SUPPORTED_MOODS: List[str] = [
    "happy",
    "sad",
    "romantic",
    "energetic",
    "tense",
    "dark",
    "excited",
    "calm",
]

MOOD_VISUALS: Dict[str, Dict[str, str]] = {
    "happy": {"color": "#ffe066", "emoji": "😄"},
    "sad": {"color": "#74c0fc", "emoji": "😢"},
    "romantic": {"color": "#ff8787", "emoji": "❤️"},
    "energetic": {"color": "#ff6b6b", "emoji": "🔥"},
    "tense": {"color": "#b197fc", "emoji": "😬"},
    "dark": {"color": "#868e96", "emoji": "🖤"},
    "calm": {"color": "#96f2d7", "emoji": "😌"},
    "excited": {"color": "#ffa94d", "emoji": "🤩"},
}

DEFAULT_BG = "#1e1e24"
DEFAULT_FG = "#f8f9fa"


def load_artifacts():
    missing = []
    for path, label in [
        (MODEL_PATH, "Model"),
        (VECTORIZER_PATH, "Vectorizer"),
        (LABEL_ENCODER_PATH, "Label encoder"),
    ]:
        if not os.path.exists(path):
            missing.append(f"{label} ({os.path.basename(path)})")

    if missing:
        messagebox.showerror(
            "CINEVI Startup Error",
            "Required model artifacts are missing:\n\n"
            + "\n".join(missing)
            + "\n\nPlease run 'cinevi_train.py' first.",
        )
        sys.exit(1)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    return model, vectorizer, label_encoder


def load_labeled_movies(csv_path: str = LABELED_MOVIES_CSV) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        messagebox.showerror(
            "CINEVI Startup Error",
            f"Labeled movie dataset not found at:\n{csv_path}\n\n"
            "Please run 'cinevi_train.py' to generate it.",
        )
        sys.exit(1)

    df = pd.read_csv(csv_path)
    expected_cols = {"title", "genres", "mood"}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        messagebox.showerror(
            "CINEVI Dataset Error",
            "Labeled dataset is missing required columns:\n"
            + ", ".join(sorted(missing_cols)),
        )
        sys.exit(1)

    return df


def normalize_user_mood(text: str) -> str:
    cleaned = text.strip().lower()
    if not cleaned:
        return ""

    # Direct match first
    if cleaned in SUPPORTED_MOODS:
        return cleaned

    # Keyword-based mapping for simple emotional phrases
    keyword_map: List[Tuple[List[str], str]] = [
        (["happy", "joy", "cheer", "smile"], "happy"),
        (["sad", "down", "blue", "cry"], "sad"),
        (["love", "romantic", "date", "valentine"], "romantic"),
        (["energy", "energetic", "pumped", "action"], "energetic"),
        (["tense", "anxious", "stress", "edge"], "tense"),
        (["dark", "gothic", "grim", "horror"], "dark"),
        (["excited", "hype", "thrill"], "excited"),
        (["relax", "calm", "chill", "peace"], "calm"),
    ]

    for keywords, mood in keyword_map:
        if any(word in cleaned for word in keywords):
            return mood

    return "calm"


class CineviApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CINEVI — Movie Mood AI")
        self.root.geometry("460x420")
        self.root.resizable(False, False)

        self.root.configure(bg=DEFAULT_BG)

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "TButton",
            font=("Segoe UI", 11, "bold"),
            padding=6,
        )
        style.configure(
            "TEntry",
            padding=4,
        )

        self.model, self.vectorizer, self.label_encoder = load_artifacts()
        self.movies_df = load_labeled_movies()
        self.last_recommendations: List[str] = []

        self._build_widgets()

    def _build_widgets(self) -> None:
        title = tk.Label(
            self.root,
            text="CINEVI",
            font=("Segoe UI", 20, "bold"),
            fg=DEFAULT_FG,
            bg=DEFAULT_BG,
        )
        title.pack(pady=(16, 4))

        subtitle = tk.Label(
            self.root,
            text="Movie Mood AI — from feelings to films",
            font=("Segoe UI", 10),
            fg="#ced4da",
            bg=DEFAULT_BG,
        )
        subtitle.pack(pady=(0, 16))

        input_frame = tk.Frame(self.root, bg=DEFAULT_BG)
        input_frame.pack(pady=(0, 10), padx=16, fill="x")

        prompt = tk.Label(
            input_frame,
            text="How do you feel?",
            font=("Segoe UI", 11),
            fg=DEFAULT_FG,
            bg=DEFAULT_BG,
        )
        prompt.pack(anchor="w")

        self.mood_var = tk.StringVar()
        entry = ttk.Entry(input_frame, textvariable=self.mood_var, font=("Segoe UI", 11))
        entry.pack(side="left", fill="x", expand=True, pady=(4, 0))
        entry.bind("<Return>", lambda event: self.on_recommend())

        recommend_btn = ttk.Button(
            input_frame,
            text="Recommend Movies",
            command=self.on_recommend,
        )
        recommend_btn.pack(side="left", padx=(8, 0), pady=(4, 0))

        status_frame = tk.Frame(self.root, bg=DEFAULT_BG)
        status_frame.pack(pady=(8, 4), padx=16, fill="x")

        self.mood_label = tk.Label(
            status_frame,
            text="Detected mood: —",
            font=("Segoe UI", 11, "bold"),
            fg=DEFAULT_FG,
            bg=DEFAULT_BG,
        )
        self.mood_label.pack(side="left")

        self.emoji_label = tk.Label(
            status_frame,
            text="",
            font=("Segoe UI", 18),
            fg=DEFAULT_FG,
            bg=DEFAULT_BG,
        )
        self.emoji_label.pack(side="right")

        results_frame = tk.Frame(self.root, bg=DEFAULT_BG)
        results_frame.pack(pady=(8, 16), padx=16, fill="both", expand=True)

        self.results_container = tk.Frame(
            results_frame,
            bg="#212529",
            highlightbackground="#343a40",
            highlightthickness=1,
        )
        self.results_container.pack(fill="both", expand=True)

        header = tk.Label(
            self.results_container,
            text="Recommended Movies",
            font=("Segoe UI", 11, "bold"),
            fg=DEFAULT_FG,
            bg="#212529",
        )
        header.pack(anchor="w", padx=10, pady=(8, 4))

        self.results_text = tk.Text(
            self.results_container,
            height=10,
            wrap="word",
            bg="#212529",
            fg=DEFAULT_FG,
            relief="flat",
            font=("Segoe UI", 10),
        )
        self.results_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.results_text.configure(state="disabled")

        self.results_text.configure(cursor="arrow")

    def on_recommend(self) -> None:
        user_input = self.mood_var.get()
        normalized_mood = normalize_user_mood(user_input)

        if not normalized_mood:
            messagebox.showwarning(
                "Input required", "Please enter how you feel to get recommendations."
            )
            return

        mood_info = MOOD_VISUALS.get(normalized_mood, MOOD_VISUALS["calm"])
        detected_text = f"Detected mood: {normalized_mood.capitalize()}"
        self.mood_label.configure(text=detected_text)
        self.emoji_label.configure(text=mood_info["emoji"])

        self.root.configure(bg=mood_info["color"])
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Frame):
                widget.configure(bg=mood_info["color"])
            if isinstance(widget, tk.Label):
                widget.configure(bg=mood_info["color"])

        self.results_container.configure(bg=_darken_color(mood_info["color"], 0.85))

        recs = self._get_recommendations(normalized_mood, n=5)

        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", tk.END)
        if not recs:
            self.results_text.insert(
                tk.END,
                "No movies found for this mood. Try a different feeling or update the dataset.",
            )
        else:
            for i, title in enumerate(recs, start=1):
                self.results_text.insert(tk.END, f"{i}. {title}\n")
        self.results_text.configure(state="disabled")

    def _get_recommendations(self, mood: str, n: int = 5) -> List[str]:
        subset = self.movies_df[self.movies_df["mood"] == mood]
        titles = subset["title"].dropna().tolist()
        if not titles:
            return []

        if len(titles) <= n:
            recs = titles.copy()
            random.shuffle(recs)
        else:
            recs = random.sample(titles, n)

        if recs == self.last_recommendations and len(titles) > n:
            all_sets = [
                list(s)
                for s in {
                    tuple(sorted(random.sample(titles, n))) for _ in range(10)
                }
            ]
            for candidate in all_sets:
                if candidate != self.last_recommendations:
                    recs = candidate
                    break

        self.last_recommendations = recs
        return recs


def _darken_color(hex_color: str, factor: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "#212529"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def main() -> None:
    root = tk.Tk()
    app = CineviApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

