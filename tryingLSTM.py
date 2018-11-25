import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
import matplotlib.pyplot as pyplot




def clean_text(text):
    # print("Text:")
    # print(text)
    # Remove puncuation
    # text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = text.split()
    # remove punctuation from each token
    table = str.maketrans ( '' , '' , string.punctuation )
    tokens = [ w.translate ( table ) for w in tokens ]
    tokens = [ word for word in tokens if word.isalpha () ]

    # Convert words to lower case and split them
    # text = text.lower().split()


    # Remove stop words
    # stops = set(stopwords.words("english"))
    # text = [w for w in text if not w in stops and len(w) >= 3]

    # filter out stop words
    stop_words = set ( stopwords.words ( 'english' ) )
    tokens = [ w for w in tokens if not w in stop_words ]
    # filter out short tokens
    tokens = [ word for word in tokens if len ( word ) > 1 ]

    text = " ".join(tokens)
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
    # stemmer = PorterStemmer()
    # stemmed_words = [stemmer.stem(word) for word in text]
    # text = " ".join(stemmed_words)

    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_words = [wordnet_lemmatizer.lemmatize(word) for word in text]
    text = " ".join(lemmatized_words)

    return text


def row_to_vec(row):
    if row['Label'] == 'positive':
        return 1
    if row['Label'] == 'negative':
        return 0


def get_data():
    reviews_df = pd.read_csv("data.csv", error_bad_lines=False)
    reviews_df = reviews_df.loc[(reviews_df['Label'] == 'positive') | (reviews_df['Label'] == 'negative')]

    reviews_df['Numeric_Label'] = reviews_df.apply(row_to_vec, axis=1)

    reviews_df[ "Review" ] = reviews_df[ 'Title' ].astype ( str ) + ' ' + reviews_df[ 'Description' ]

    reviews_df[ 'Review' ] = reviews_df[ 'Review' ].map ( lambda x: clean_text ( x ) )

    print(reviews_df)
    return reviews_df


def pre_process_data(data):
    X , y = (data[ 'Review' ].values , data[ 'Numeric_Label' ].values)

    tk = Tokenizer ( lower=True )
    tk.fit_on_texts ( X )
    X_seq = tk.texts_to_sequences ( X )
    X_pad = pad_sequences ( X_seq , maxlen=100 , padding='post' )
    print(X_pad,y)
    return X_pad, y, tk


def train_model(x , y, tk):
    X_train , X_test , y_train , y_test = train_test_split ( x , y , test_size=0.25 , random_state=1 )

    batch_size = 4
    X_train1 = X_train[ batch_size: ]
    y_train1 = y_train[ batch_size: ]
    X_valid = X_train[ :batch_size ]
    y_valid = y_train[ :batch_size ]

    vocabulary_size = len ( tk.word_counts.keys () ) + 1
    max_words = 100
    embedding_size = 128
    model = Sequential ()
    model.add ( Embedding ( vocabulary_size , embedding_size , input_length=max_words ) )
    model.add ( SpatialDropout1D(0.4))
    model.add ( LSTM ( 200, dropout=0.2, recurrent_dropout=0.2 ) )
    model.add ( Dense ( 1 , activation='sigmoid' ) )
    model.compile ( loss='binary_crossentropy' , optimizer='adam' , metrics=[ 'accuracy' ] )
    history = model.fit(X_train1, y_train1, validation_data= (X_valid, y_valid), batch_size= batch_size, epochs=15)

    pyplot.plot ( history.history[ 'loss' ] )
    pyplot.plot ( history.history[ 'val_loss' ] )
    pyplot.title ( 'model train vs validation loss' )
    pyplot.ylabel ( 'loss' )
    pyplot.xlabel ( 'epoch' )
    pyplot.legend ( [ 'train' , 'validation' ] , loc='upper right' )
    pyplot.show ()

    scores = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy = ', scores[1])
    pass


def main():
    df = get_data()
    x,y,tk = pre_process_data(df)
    train_model(x,y,tk)


if __name__ == '__main__':
    main()