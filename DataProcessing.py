import re
import string

import pandas as pd
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


def row_to_vec(row):
    if row['Label'] == 'positive':
        return 1
    if row['Label'] == 'negative':
        return 0


# Text Normalizing function. Part of the following function was taken from this link.
def clean_text(text):
    # print("Text:")
    # print(text)
    # Remove puncuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Convert words to lower case and split them
    text = text.lower().split()

    # Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Stemming
    text = text.split()
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    # print("Cleaned text:")
    # print(text)
    return text


def get_data():
    reviews_df = pd.read_csv("data.csv", error_bad_lines=False)

    reviews_df = reviews_df.loc[(reviews_df['Label'] == 'positive') | (reviews_df['Label'] == 'negative')]
    # print(len(reviews_df.loc[(reviews_df['Label'] == 'positive')]))
    # print(len(reviews_df.loc[(reviews_df['Label'] == 'negative')]))
    reviews_df['Numeric_Label'] = reviews_df.apply(row_to_vec, axis=1)

    reviews_df["Review"] = reviews_df['Title'].astype(str) + ' ' + reviews_df['Description']

    reviews_df['Review'] = reviews_df['Review'].map(lambda x: clean_text(x))

    return reviews_df


def get_encoded_padded_content(tokenizer, content, max_length):
    # integer encode the documents
    encoded_docs = tokenizer.texts_to_sequences(content)

    # pad documents to a max length of 4 words
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    # print(padded_docs)
    return padded_docs


def main():
    reviews_df = get_data()
    print(reviews_df)
    # tokenize all content
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(reviews_df['Review'])
    num_encoder_tokens = len(tokenizer.word_index) + 1

    # create training and testing vars
    X_train, X_test, y_train, y_test = train_test_split(reviews_df['Review'], reviews_df['Numeric_Label'],
                                                        test_size=0.2, shuffle=True)

    max_review_length = 500
    X_train = get_encoded_padded_content(tokenizer, X_train, max_review_length)
    X_test = get_encoded_padded_content(tokenizer, X_test, max_review_length)
    # print(X_train)
    # print(y_train)
    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(num_encoder_tokens, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model_history = model.fit(X_train, y_train, epochs=3, batch_size=4, validation_split=0.2)
    print(model_history.history)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    pass


if __name__ == '__main__':
    main()
