import pandas as pd
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as pyplot


def clean_text(text):

    # Split into tokens.
    # Convert to lowercase.
    # Remove punctuation from each token.
    # Filter out remaining tokens that are not alphabetic.
    # Filter out tokens that are stop words.

    # split into words
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize ( text )
    # convert to lower case
    tokens = [ w.lower () for w in tokens ]
    # remove punctuation from each word
    import string
    table = str.maketrans ( '' , '' , string.punctuation )
    stripped = [ w.translate ( table ) for w in tokens ]
    # remove remaining tokens that are not alphabetic
    words = [ word for word in stripped if word.isalpha () ]
    # filter out stop words
    from nltk.corpus import stopwords
    stop_words = set ( stopwords.words ( 'english' ) )
    words = [ w for w in words if not w in stop_words ]

    wordnet_lemmatizer = WordNetLemmatizer ()
    lemmatized_words = [ wordnet_lemmatizer.lemmatize ( word ) for word in words ]
    new_text = " ".join ( lemmatized_words )


    return new_text


def row_to_vec(row):
    if row['Label'] == 'positive':
        return 0
    if row['Label'] == 'negative':
        return 1


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

    batch_size = 64
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
    history = model.fit(X_train1, y_train1, validation_data= (X_valid, y_valid), batch_size= batch_size, epochs=7)

    pyplot.plot ( history.history[ 'loss' ] )
    pyplot.plot ( history.history[ 'val_loss' ] )
    pyplot.title ( 'model train vs validation loss' )
    pyplot.ylabel ( 'loss' )
    pyplot.xlabel ( 'epoch' )
    pyplot.legend ( [ 'train' , 'validation' ] , loc='upper right' )
    pyplot.show ()

    scores = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy = ', scores[1])
    y_pred = model.predict ( X_test )
    y_pred = (y_pred > 0.5)
    print('Confusion Matrix:')
    cm = confusion_matrix ( y_test , y_pred )
    print(cm)
    pass


def main():
    df = get_data()
    x,y,tk = pre_process_data(df)
    train_model(x,y,tk)


if __name__ == '__main__':
    main()