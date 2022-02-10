import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
with open('intents.json', 'r') as f:
    intents = json.load(f)
words = []
classes = []
documents = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        documents.append((word, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
ignore_letters = ['!', '?', ',', '.']
lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))

classes_dict = {classes[i]: i for i in range(len(classes))}

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
for doc in documents:
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    bag = [1 if word in word_patterns else 0 for word in words]
    target = classes_dict[doc[1]]
    training.append([bag, target])

random.shuffle(training)
X = [pattern[0] for pattern in training]
y = [pattern[1] for pattern in training]

model = keras.models.Sequential([
    keras.layers.Dense(128, input_shape=[len(X[0])], activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, epochs=400, batch_size=4, verbose=1)

classes = []
documents = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        documents.append((pattern, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

random.shuffle(documents)
X = [pattern[0] for pattern in documents]
y = [pattern[1] for pattern in documents]


vocab_size = 100
max_len = 20
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X)


X_tokenized = tokenizer.texts_to_sequences(X)
X_tokenized = keras.preprocessing.sequence.pad_sequences(X_tokenized, maxlen=max_len)


y = np.array([classes_dict[tag] for tag in y])
embedding_dim = 20



model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_shape=[None]),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.LSTM(32),
    keras.layers.Dense(len(classes), activation='softmax')
])


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_tokenized, y, epochs=400, batch_size=4, verbose=1)





classes = []
documents = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        documents.append((pattern, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

random.shuffle(documents)
X = [pattern[0] for pattern in documents]
y = [pattern[1] for pattern in documents]





from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

lencoder= LabelEncoder()
y = lencoder.fit_transform(y)




pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
pickle.dump(lencoder, open('lencoder.pkl', 'wb'))



model = keras.models.Sequential([
    keras.layers.Dense(128, input_shape=[len(vectorizer.get_feature_names())], activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(classes), activation='softmax')
])


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X.todense(), y, epochs=400, batch_size=4, verbose=1)

model.save('chatbot_model.h5', model)





