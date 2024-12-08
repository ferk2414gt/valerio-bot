#Entrenamiento
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# Keras para crear la red neuronal
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

# Inicializamos el lematizador   
lemmatizer = WordNetLemmatizer()

# Cargamos el archivo intents.json
intents = json.loads(open('intents.json').read())

# Descargar recursos de NLTK necesarios
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Variables para almacenar palabras, clases y documentos
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']

# Procesamos los patrones y etiquetas de intents.json
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lematizamos y eliminamos duplicados
words = sorted(set([lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]))
classes = sorted(classes)

# Guardamos palabras y clases como archivos pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Preparación para entrenamiento: transformar en "bolsa de palabras"
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in document[0]]
    bag = [1 if word in word_patterns else 0 for word in words]
    
    output_row = output_empty[:]
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Mezclamos y dividimos los datos en entradas y salidas
random.shuffle(training)
training = np.array(training, dtype=object)
print(training)


#dividioms en dos variables 

train_x = list(training[:,0])
train_y = list(training[:,1])

#modelo 

model = Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

train_process = model.fit(np.array(train_x),np.array(train_y), epochs=100, batch_size=5, verbose=1)
model.save("pruebafff_chat.h5",train_process)