#chatbot
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# Keras para crear la red neuronal
from keras.models import load_model

# Inicializamos el lematizador   
lemmatizer = WordNetLemmatizer()

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

intents = json.loads(open('n.json').read())

words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))
model=load_model('pruebafff_chat.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]  
    return category

def get_response(tag,intents_json):
    list_of_intents = intents_json['intents']   
    result = ""
    for i in list_of_intents:
        if i["tag"]==tag:
           result = random.choice(i['responses'])
           break
    return result              


