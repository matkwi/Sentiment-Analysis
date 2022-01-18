from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import random

import pandas as pd
import re
import string
import time

import numpy as np

import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.metrics import plot_confusion_matrix


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def load_dataset(filepath, cols):
    df = pd.read_csv(filepath, encoding='latin-1')
    df.columns = cols
    return df

def delete_unwanted_cols(df, cols):
    for col in cols:
        del df[col]
    return df

def preprocess_tweet_text(tweet):
    # text to lowercase
    tweet = tweet.lower()

    # remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)

    # remove punctuations
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))

    # remove @ references and #
    tweet = re.sub(r'\@\w+|\#', "", tweet)

    # remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [word for word in tweet_tokens if word not in stop_words]

    # stemming
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in filtered_words]

    # lemmatizing
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]

    return " ".join(lemma_words)

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

# Loading datasets
columns = ['polarity', 'id', 'timestamp', 'query', 'user', 'tweet']
dataset = load_dataset('drive/MyDrive/data/training.1600000.processed.noemoticon.csv', columns)
print("Collected")

from google.colab import drive
drive._mount('/content/drive')

# Getting only 6000 tweets from whole data
top = dataset.head(3000)
bottom = dataset.tail(3000)
train_dataset = pd.concat([top, bottom])
train_dataset.reset_index(inplace=True, drop=True)

# deleting redundant columns
del_cols = ['timestamp', 'query', 'user']
train_dataset = delete_unwanted_cols(train_dataset, del_cols)

not_shuffled_d = []
not_shuffled_l = []
for i in range(len(train_dataset)):
    not_shuffled_d.append(train_dataset['tweet'][i])
    if train_dataset['polarity'][i] == 4:
        not_shuffled_l.append(1)
    else:
        not_shuffled_l.append(0)

print("created dataset")

split1_data = []
split1_data_labels = []
randPick = list(range(0, len(not_shuffled_d)))
random.shuffle(randPick)
for x in randPick:
    split1_data.append(not_shuffled_d[x])
    split1_data_labels.append(not_shuffled_l[x])

print("shuffled dataset")

vectorizer = CountVectorizer(
    analyzer='word',
    lowercase=False,
)
split1_features = vectorizer.fit_transform(
    split1_data
)
split1_features_nd = split1_features.toarray()

split1_X_train, split1_X_test, split1_y_train, split1_y_test = train_test_split(
    split1_features_nd,
    split1_data_labels,
    train_size=0.90,
    random_state=1234
)

split1_X_train, split1_X_val, split1_y_train, split1_y_val = train_test_split(
    split1_X_train,
    split1_y_train,
    train_size=0.89,
    random_state=1234
)

print("split1 done")

split3_data = []
split3_data_labels = []
# Data preprocessing
for i in range(len(split1_data)):
    split3_data.append(preprocess_tweet_text(split1_data[i]))
    split3_data_labels.append(split1_data_labels[i])
print("Preprocessed")

split3_features = vectorizer.fit_transform(
    split3_data
)
split3_features_nd = split3_features.toarray()

split3_X_train, split3_X_test, split3_y_train, split3_y_test = train_test_split(
    split3_features_nd,
    split3_data_labels,
    train_size=0.80,
    random_state=1234
)

split2_X_train = split3_X_train.copy()
split2_X_test = split3_X_test.copy()
split2_y_train = split3_y_train.copy()
split2_y_test = split3_y_test.copy()

split2_X_train, split2_X_val, split2_y_train, split2_y_val = train_test_split(
    split2_X_train,
    split2_y_train,
    train_size=0.89,
    random_state=1234
)

split3_X_val = split2_X_val.copy()
split3_y_val = split2_y_val.copy()

print("split 2 and 3 done")

split1_X_val = np.array(split1_X_val)
split1_y_val = np.array(split1_y_val)
split1_X_train = np.array(split1_X_train)
split1_y_train = np.array(split1_y_train)
split1_X_test = np.array(split1_X_test)
split1_y_test = np.array(split1_y_test)

split2_X_val = np.array(split2_X_val)
split2_y_val = np.array(split2_y_val)
split2_X_train = np.array(split2_X_train)
split2_y_train = np.array(split2_y_train)
split2_X_test = np.array(split2_X_test)
split2_y_test = np.array(split2_y_test)

split3_X_val = np.array(split3_X_val)
split3_y_val = np.array(split3_y_val)
split3_X_train = np.array(split3_X_train)
split3_y_train = np.array(split3_y_train)
split3_X_test = np.array(split3_X_test)
split3_y_test = np.array(split3_y_test)

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

n_words = split1_X_test.shape[1]
# define network
model1 = Sequential()
model1.add(Dense(50, input_shape=(n_words,), activation='relu'))
model1.add(Dense(1, activation='relu'))

# compile network
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m, precision_m, recall_m])

errors_train = []
errors_val = []
errors_test = []

acc_train = []
acc_val = []
acc_test = []

f1_train = []
f1_val = []
f1_test = []

prec_train = []
prec_val = []
prec_test = []

rec_train = []
rec_val = []
rec_test = []

start = time.time()
for i in range(30):
  model1.fit(split1_X_train, split1_y_train, epochs=1, verbose=0, validation_data = (split1_X_val, split1_y_val))
  loss, acc, f1_score, precision, recall = model1.evaluate(split1_X_val, split1_y_val, verbose=0)
  errors_val.append(loss)
  acc_val.append(acc)
  f1_val.append(f1_score)
  prec_val.append(precision)
  rec_val.append(recall)
  loss, acc, f1_score, precision, recall = model1.evaluate(split1_X_train, split1_y_train, verbose=0)
  errors_train.append(loss)
  acc_train.append(acc)
  f1_train.append(f1_score)
  prec_train.append(precision)
  rec_train.append(recall)
  loss, acc, f1_score, precision, recall = model1.evaluate(split1_X_test, split1_y_test, verbose=0)
  errors_test.append(loss)
  acc_test.append(acc)
  f1_test.append(f1_score)
  prec_test.append(precision)
  rec_test.append(recall)

end = time.time()
print("time: ", end - start)

p = plt.plot(range(len(errors_val)), errors_val, color='blue', linewidth=1.5, label="val")
p = plt.plot(range(len(errors_train)), errors_train, color='red', linewidth=1.5, label="train")
p = plt.plot(range(len(errors_test)), errors_test, color='green', linewidth=1.5, label="test")
p = plt.legend(loc="upper right")
plt.show(p)

p1 = plt.plot(range(len(acc_val)), acc_val, color='blue', linewidth=1.5, label="val")
p1 = plt.plot(range(len(acc_train)), acc_train, color='red', linewidth=1.5, label="train")
p1 = plt.plot(range(len(acc_test)), acc_test, color='green', linewidth=1.5, label="test")
p1 = plt.legend(loc="upper right")
plt.show(p1)

p2 = plt.plot(range(len(f1_val)), f1_val, color='blue', linewidth=1.5, label="val")
p2 = plt.plot(range(len(f1_train)), f1_train, color='red', linewidth=1.5, label="train")
p2 = plt.plot(range(len(f1_test)), f1_test, color='green', linewidth=1.5, label="test")
p2 = plt.legend(loc="upper right")
plt.show(p2)

p3 = plt.plot(range(len(prec_val)), prec_val, color='blue', linewidth=1.5, label="val")
p3 = plt.plot(range(len(prec_train)), prec_train, color='red', linewidth=1.5, label="train")
p3 = plt.plot(range(len(prec_test)), prec_test, color='green', linewidth=1.5, label="test")
p3 = plt.legend(loc="upper right")
plt.show(p3)

p4 = plt.plot(range(len(rec_val)), rec_val, color='blue', linewidth=1.5, label="val")
p4 = plt.plot(range(len(rec_train)), rec_train, color='red', linewidth=1.5, label="train")
p4 = plt.plot(range(len(rec_test)), rec_test, color='green', linewidth=1.5, label="test")
p4 = plt.legend(loc="upper right")
plt.show(p4)

from google.colab import drive
drive._mount('/content/drive')

n_words = split2_X_test.shape[1]
# define network
model2 = Sequential()
model2.add(Dense(100, input_shape=(n_words,), activation='relu'))
model2.add(Dense(1, activation='sigmoid'))

# compile network
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m, precision_m, recall_m])
import keras.backend as K
model2.optimizer.lr = 0.001

# fit network
errors_train = []
errors_val = []
errors_test = []

acc_train = []
acc_val = []
acc_test = []

f1_train = []
f1_val = []
f1_test = []

prec_train = []
prec_val = []
prec_test = []

rec_train = []
rec_val = []
rec_test = []
start = time.time()
for i in range(30):
  model2.fit(split2_X_train, split2_y_train, epochs=1, verbose=0, validation_data = (split2_X_val, split2_y_val))
  loss, acc, f1_score, precision, recall = model2.evaluate(split2_X_val, split2_y_val, verbose=0)
  errors_val.append(loss)
  acc_val.append(acc)
  f1_val.append(f1_score)
  prec_val.append(precision)
  rec_val.append(recall)
  loss, acc, f1_score, precision, recall = model2.evaluate(split2_X_train, split2_y_train, verbose=0)
  errors_train.append(loss)
  acc_train.append(acc)
  f1_train.append(f1_score)
  prec_train.append(precision)
  rec_train.append(recall)
  loss, acc, f1_score, precision, recall = model2.evaluate(split2_X_test, split2_y_test, verbose=0)
  errors_test.append(loss)
  acc_test.append(acc)
  f1_test.append(f1_score)
  prec_test.append(precision)
  rec_test.append(recall)
end = time.time()
print("time: ", end - start)
plt.plot(range(len(errors_val)), errors_val, color='blue', linewidth=1.5, label="val")
plt.plot(range(len(errors_train)), errors_train, color='red', linewidth=1.5, label="train")
plt.legend(loc="upper right")


p = plt.plot(range(len(errors_val)), errors_val, color='blue', linewidth=1.5, label="val")
p = plt.plot(range(len(errors_train)), errors_train, color='red', linewidth=1.5, label="train")
p = plt.plot(range(len(errors_test)), errors_test, color='green', linewidth=1.5, label="test")
p = plt.legend(loc="upper right")
plt.show(p)

p1 = plt.plot(range(len(acc_val)), acc_val, color='blue', linewidth=1.5, label="val")
p1 = plt.plot(range(len(acc_train)), acc_train, color='red', linewidth=1.5, label="train")
p1 = plt.plot(range(len(acc_test)), acc_test, color='green', linewidth=1.5, label="test")
p1 = plt.legend(loc="upper right")
plt.show(p1)

p2 = plt.plot(range(len(f1_val)), f1_val, color='blue', linewidth=1.5, label="val")
p2 = plt.plot(range(len(f1_train)), f1_train, color='red', linewidth=1.5, label="train")
p2 = plt.plot(range(len(f1_test)), f1_test, color='green', linewidth=1.5, label="test")
p2 = plt.legend(loc="upper right")
plt.show(p2)

p3 = plt.plot(range(len(prec_val)), prec_val, color='blue', linewidth=1.5, label="val")
p3 = plt.plot(range(len(prec_train)), prec_train, color='red', linewidth=1.5, label="train")
p3 = plt.plot(range(len(prec_test)), prec_test, color='green', linewidth=1.5, label="test")
p3 = plt.legend(loc="upper right")
plt.show(p3)

p4 = plt.plot(range(len(rec_val)), rec_val, color='blue', linewidth=1.5, label="val")
p4 = plt.plot(range(len(rec_train)), rec_train, color='red', linewidth=1.5, label="train")
p4 = plt.plot(range(len(rec_test)), rec_test, color='green', linewidth=1.5, label="test")
p4 = plt.legend(loc="upper right")
plt.show(p4)

n_words = split3_X_test.shape[1]
# define network
model3 = Sequential()
model3.add(Dense(50, input_shape=(n_words,), activation='relu'))
model3.add(Dense(1, activation='sigmoid'))

# compile network
model3.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy', f1_m, precision_m, recall_m])

# fit network
print(K.eval(model1.optimizer.lr))
errors_train = []
errors_val = []
errors_test = []

acc_train = []
acc_val = []
acc_test = []

f1_train = []
f1_val = []
f1_test = []

prec_train = []
prec_val = []
prec_test = []

rec_train = []
rec_val = []
rec_test = []
start = time.time()
for i in range(200):
  model3.fit(split3_X_train, split3_y_train, epochs=1, verbose=0, validation_data = (split3_X_val, split3_y_val))
  loss, acc, f1_score, precision, recall = model3.evaluate(split3_X_val, split3_y_val, verbose=0)
  errors_val.append(loss)
  acc_val.append(acc)
  f1_val.append(f1_score)
  prec_val.append(precision)
  rec_val.append(recall)
  loss, acc, f1_score, precision, recall = model3.evaluate(split3_X_train, split3_y_train, verbose=0)
  errors_train.append(loss)
  acc_train.append(acc)
  f1_train.append(f1_score)
  prec_train.append(precision)
  rec_train.append(recall)
  loss, acc, f1_score, precision, recall = model3.evaluate(split3_X_test, split3_y_test, verbose=0)
  errors_test.append(loss)
  acc_test.append(acc)
  f1_test.append(f1_score)
  prec_test.append(precision)
  rec_test.append(recall)
end = time.time()
print("time: ", end - start)
plt.plot(range(len(errors_val)), errors_val, color='blue', linewidth=1.5, label="val")
plt.plot(range(len(errors_train)), errors_train, color='red', linewidth=1.5, label="train")
plt.legend(loc="upper right")


p = plt.plot(range(len(errors_val)), errors_val, color='blue', linewidth=1.5, label="val")
p = plt.plot(range(len(errors_train)), errors_train, color='red', linewidth=1.5, label="train")
p = plt.plot(range(len(errors_test)), errors_test, color='green', linewidth=1.5, label="test")
p = plt.legend(loc="upper right")
plt.show(p)

p1 = plt.plot(range(len(acc_val)), acc_val, color='blue', linewidth=1.5, label="val")
p1 = plt.plot(range(len(acc_train)), acc_train, color='red', linewidth=1.5, label="train")
p1 = plt.plot(range(len(acc_test)), acc_test, color='green', linewidth=1.5, label="test")
p1 = plt.legend(loc="upper right")
plt.show(p1)

p2 = plt.plot(range(len(f1_val)), f1_val, color='blue', linewidth=1.5, label="val")
p2 = plt.plot(range(len(f1_train)), f1_train, color='red', linewidth=1.5, label="train")
p2 = plt.plot(range(len(f1_test)), f1_test, color='green', linewidth=1.5, label="test")
p2 = plt.legend(loc="upper right")
plt.show(p2)

p3 = plt.plot(range(len(prec_val)), prec_val, color='blue', linewidth=1.5, label="val")
p3 = plt.plot(range(len(prec_train)), prec_train, color='red', linewidth=1.5, label="train")
p3 = plt.plot(range(len(prec_test)), prec_test, color='green', linewidth=1.5, label="test")
p3 = plt.legend(loc="upper right")
plt.show(p3)

p4 = plt.plot(range(len(rec_val)), rec_val, color='blue', linewidth=1.5, label="val")
p4 = plt.plot(range(len(rec_train)), rec_train, color='red', linewidth=1.5, label="train")
p4 = plt.plot(range(len(rec_test)), rec_test, color='green', linewidth=1.5, label="test")
p4 = plt.legend(loc="upper right")
plt.show(p4)