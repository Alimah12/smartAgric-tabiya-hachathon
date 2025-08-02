""" This file contains the code for the chatbot response. """

import os
import json
import pickle
import random

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from deep_translator import GoogleTranslator
from flask import Flask
from flask_socketio import SocketIO, emit

# ---- NLTK data setup ----
# If you bundle your own nltk_data dir, uncomment and adjust:
# nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))

for pkg in ("punkt", "punkt_tab", "wordnet"):
    try:
        nltk.data.find(f"tokenizers/{pkg}") if pkg.startswith("punkt") else nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, quiet=True)

# ---- Load model artifacts ----
BASE_DIR = os.path.dirname(__file__)
model = load_model(os.path.join(BASE_DIR, "model.h5"))

with open(os.path.join(BASE_DIR, "intents.json"), encoding="utf-8") as f:
    intents = json.load(f)

with open(os.path.join(BASE_DIR, "word.pkl"), "rb") as f:
    words = pickle.load(f)

with open(os.path.join(BASE_DIR, "class.pkl"), "rb") as f:
    classes = pickle.load(f)

lemma = WordNetLemmatizer()

# ---- NLP utility functions ----

def clean_up_sentence(sentence: str) -> list[str]:
    """Tokenize and lemmatize an incoming sentence."""
    tokens = nltk.word_tokenize(sentence)
    return [lemma.lemmatize(tok.lower()) for tok in tokens]

def bow(sentence: str, vocabulary: list[str], show_details: bool = False) -> np.ndarray:
    """Convert a sentence into a bag-of-words numpy array."""
    sentence_words = clean_up_sentence(sentence)
    bag = np.zeros(len(vocabulary), dtype=np.float32)
    for w in sentence_words:
        for idx, vocab_word in enumerate(vocabulary):
            if vocab_word == w:
                bag[idx] = 1
                if show_details:
                    print(f"Found '{w}' in bag")
    return bag

def predict_class(sentence: str) -> list[dict]:
    """Predict the intent class of a sentence using the loaded model."""
    bow_vec = bow(sentence, words, show_details=False)
    probabilities = model.predict(np.array([bow_vec]))[0]
    ERROR_THRESHOLD = 0.25

    results = [
        (i, prob) for i, prob in enumerate(probabilities)
        if prob > ERROR_THRESHOLD
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return [
        {"intent": classes[idx], "probability": str(prob)}
        for idx, prob in results
    ]

def get_response(ints: list[dict], intents_json: dict) -> str:
    """Select a random response from the intents file."""
    if not ints:
        return "Sorry, I didn't understand that."
    tag = ints[0]["intent"]
    for intent_block in intents_json["intents"]:
        if intent_block["tag"] == tag:
            return random.choice(intent_block["responses"])
    return "Sorry, I didn't understand that."

# ---- Translation helper ----

def translate_message(
    message: str,
    source_language: str,
    target_language: str = "en"
) -> str:
    """Translate a message using GoogleTranslator; fall back on original if error."""
    try:
        return GoogleTranslator(
            source=source_language,
            target=target_language
        ).translate(message)
    except Exception as e:
        print(f"Translation error: {e}")
        return message

# ---- Chatbot core ----

def chatbot_response(user_msg: str, source_language: str) -> str:
    """Full pipeline: translate → classify → respond → re-translate."""
    # 1. Translate incoming user text to English
    en_msg = translate_message(user_msg, source_language, "en")

    # 2. Predict intent and fetch an appropriate response
    predicted = predict_class(en_msg)
    reply_en = get_response(predicted, intents)

    # 3. Translate reply back to user's language
    return translate_message(reply_en, "en", source_language)

# ---- Flask‑SocketIO setup ----

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "secret!")
app.static_folder = "static"

socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on("message")
def handle_message(data):
    """Receive {'message': str, 'language': str} from client."""
    user_msg = data.get("message", "")
    lang = data.get("language", "en")
    response = chatbot_response(user_msg, lang)
    print(f"→ {response}")
    emit("recv_message", response)

if __name__ == "__main__":
    socketio.run(app, debug=True)
