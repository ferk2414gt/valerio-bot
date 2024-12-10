from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import json
import numpy as np
import random
# Cargar el modelo y configuraciones
import sys
sys.stdout.reconfigure(encoding="utf-8")
model = load_model("chatbot_lstm.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
    open("label_encoder.pkl", "rb")
    label_encoder = pickle.load("label_encoder.pkl")
with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

# Función para predecir intenciones
def predict_intent(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded)
    tag = label_encoder.inverse_transform([np.argmax(pred)])
    return tag[0]

# Función para obtener la respuesta basada en la intención
def get_response(intent, intents_json):
    for item in intents_json['intents']:
        if item['tag'] == intent:
            return random.choice(item['responses'])
    return "Lo siento, no entiendo eso."

# Cargar el archivo intents.json
with open("fotord.json",encoding="utf-8") as file:
    intents = json.load(file)

"""
# Conversación interactiva
print("Valerio Bot: ¡Hola! Soy Valerio Bot. Escribe 'salir' para finalizar la conversación.")
while True:
    user_input = input("Tú: ")  # Entrada del usuario
    if user_input.lower() == "salir":
        print("Valerio Bot: ¡Hasta luego! Fue un placer hablar contigo.")
        break
    intent = predict_intent(user_input)  # Predicción de intención
    response = get_response(intent, intents)  # Obtener la respuesta
    print(f"Valerio Bot: {response}")
"""
