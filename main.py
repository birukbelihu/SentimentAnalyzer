from tkinter import messagebox

import customtkinter as custom_tk
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

app_name = "Sentiment Analyzer"
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name)


def analyze_sentiment():
    text = entry.get().strip()

    if text == "":
        messagebox.showwarning(app_name, "Please Enter Text To Analyze")
        return

    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)

    outputs = model(**inputs)
    probs = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]
    sentiment_score = np.argmax(probs)
    confidence_score = probs[sentiment_score]
    prediction = tf.argmax(outputs.logits, axis=-1).numpy()[0]

    sentiment_map = {
        0: "Very Negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Very Positive"
    }

    color_map = {
        "Very Negative": "darkred",
        "Negative": "red",
        "Neutral": "gold",
        "Positive": "green",
        "Very Positive": "darkgreen"
    }

    sentiment = sentiment_map[prediction]

    progress_color = color_map.get(sentiment, "gray")
    progress_bar.configure(progress_color=progress_color)

    sentiment = f"Sentiment Analysis: {sentiment} \n\n Sentiment Score: {sentiment_score}"
    analysis_label.configure(text=sentiment)
    progress_bar.set(confidence_score)


custom_tk.set_appearance_mode("dark")
custom_tk.set_default_color_theme("blue")

app = custom_tk.CTk()
app.title(app_name)
app.geometry("500x500")
app.resizable(False, False)
app.iconbitmap("icon.ico")

entry = custom_tk.CTkEntry(app, width=400, placeholder_text="Enter Any Text Here")
entry.pack(pady=20)

button = custom_tk.CTkButton(app, text="Analyze Sentiment", command=analyze_sentiment)
button.pack(pady=10)

analysis_label = custom_tk.CTkLabel(app, text="Sentiment", wraplength=500, justify="left")
analysis_label.pack(pady=10)

progress_bar = custom_tk.CTkProgressBar(app, width=300)
progress_bar.set(0)
progress_bar.pack(pady=10)

app.mainloop()
