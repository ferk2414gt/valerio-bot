import random
import json
import numpy as np
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder

# Inicializamos NLTK
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')

# Cargamos el archivo JSON
with open('fotord.json') as file:
    intents = json.load(file)

# Procesamos los datos
texts = []  # Patrones
labels = []  # Etiquetas

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        texts.append(" ".join([lemmatizer.lemmatize(w.lower()) for w in word_list]))
        labels.append(intent['tag'])

# Codificamos las etiquetas
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Tokenizamos los textos
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

# Convertimos textos a secuencias y las rellenamos para igualar la longitud
sequences = tokenizer.texts_to_sequences(texts)
max_len = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Creamos un modelo LSTM
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1, output_dim=128, input_length=max_len))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compilamos el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenamos el modelo
model.fit(padded_sequences, np.array(encoded_labels), epochs=1000  , batch_size=8)

# Guardamos el modelo y las configuraciones
model.save("chatbot_lstm.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
with open("max_len.pkl", "wb") as f:
    pickle.dump(max_len, f)    
