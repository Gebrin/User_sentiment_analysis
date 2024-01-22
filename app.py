import string
from collections import Counter
import matplotlib.pyplot as plt
import streamlit as st
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


@st.cache
def process_text(text):
    lower_case = text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))


    tokenized_words = word_tokenize(cleaned_text, "english")


    final_words = [word for word in tokenized_words if word not in stopwords.words('english')]

    lemma_words = [WordNetLemmatizer().lemmatize(word) for word in final_words]

    return lemma_words

def analyze_emotions(lemma_words):
    emotion_list = []
    with open('emotions.txt', 'r') as file:
        for line in file:
            clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
            word, emotion = clear_line.split(':')

            if word in lemma_words:
                emotion_list.append(emotion)
    return emotion_list

def analyze_sentiment(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    if score['neg'] > score['pos']:
        return "Negative Sentiment"
    elif score['neg'] < score['pos']:
        return "Positive Sentiment"
    else:
        return "Neutral Sentiment"

def main():
    st.title("Sentiment Analysis App")

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        lemma_words = process_text(text)

        emotion_list = analyze_emotions(lemma_words)
        sentiment_result = analyze_sentiment(text)

        st.subheader("Emotion Analysis")
        st.write(emotion_list)

        st.subheader("Sentiment Analysis")
        st.write(sentiment_result)

        w = Counter(emotion_list)

        st.subheader("Emotion Distribution")
        fig, ax1 = plt.subplots()
        ax1.bar(w.keys(), w.values())
        fig.autofmt_xdate()
        st.pyplot(fig)

if __name__ == '__main__':
    main()
