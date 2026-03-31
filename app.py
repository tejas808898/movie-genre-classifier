import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.title("🎬 Movie Genre Classifier")

# Load dataset
data = []
with open("train_data.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.split(":::")
        if len(parts) == 4:
            genre = parts[2].strip()
            description = parts[3].strip()
            data.append([description, genre])

df = pd.DataFrame(data, columns=["description", "genre"]).dropna()

# Train model
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['description'])
y = df['genre']

model = MultinomialNB()
model.fit(X, y)

# Input
user_input = st.text_area("Enter Movie Description:")

if st.button("Predict"):
    if user_input:
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)
        st.success(f"Predicted Genre: {prediction[0]}")
    else:
        st.warning("Enter something first")
