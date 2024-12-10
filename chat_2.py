from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import json
import numpy as np
import random
import sys
sys.stdout.reconfigure(encoding="utf-8")

# Umbral de confianza
CONFIDENCE_THRESHOLD = 0.85 # 60% de confianza para que se considere una respuesta válida

# Cargar el modelo y configuraciones
model = load_model("chatbot_lstm.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

# Función para predecir intenciones
def predict_intent(text):
    seq = tokenizer.texts_to_sequences([text])  # Tokenizar el texto
    padded = pad_sequences(seq, maxlen=max_len, padding='post')  # Rellenar la secuencia
    pred = model.predict(padded)[0]  # Obtener las probabilidades de las clases
    confidence = np.max(pred)  # Obtener la probabilidad más alta

    if confidence >= CONFIDENCE_THRESHOLD:
        # Si la confianza es mayor que el umbral, devolvemos la intención predicha
        tag = label_encoder.inverse_transform([np.argmax(pred)])
        return tag[0], confidence
    else:
        # Si la confianza es baja, indicamos que no está seguro
        return "no_entendido", confidence

# Modificación de la respuesta para incluir "no sé"
def get_response(intent, intents_json, confidence):
    if intent == "no_entendido":
        return "Lo siento, no entiendo eso. ¿Puedes reformular tu pregunta?"
    
    # Si la intención es válida, devuelve una respuesta normal
    for item in intents_json['intents']:
        if item['tag'] == intent:
            return random.choice(item['responses'])

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
    
    # Predicción de intención y confianza
    intent, confidence = predict_intent(user_input)  # Predicción de intención y confianza
    
    # Obtener la respuesta basada en la intención y la confianza
    response = get_response(intent, intents, confidence)  # Ahora pasamos la confianza
    
    print(f"Valerio Bot: {response}")

   """