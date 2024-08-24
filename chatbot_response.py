import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
from keras.models import load_model
import random
import requests

# Cargar el modelo entrenado
model = load_model('chatbot_model.h5')

# Cargar los datos de palabras y clases
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Inicializar el lematizador
lemmatizer = WordNetLemmatizer()

# Cargar las intenciones
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

# Funcion para buscar en internet
def buscar_en_internet(query):
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    response = requests.get(url)

    if response.status_code == 200:
        results = response.json().get(RelatedTopics, [])
        if results:
            first_result = results[0]
            title = first_result.get("text")
            link = first_result.get("firstURL")
            return f"{title}\nMás informacion: {link}"
        else:
            return "Lo siento, no pude encontrar informacion sobre eso"
    else:
        return "Lo siento hubo un error al intentar buscar..."    
   

def clean_up_sentence(sentence):
    """Limpia y tokeniza la oración."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    """Convierte una oración en un vector de la bolsa de palabras."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def get_response(intents_list, predicted_class):
    """Obtiene una respuesta basada en la clase predicha."""
    tag = predicted_class
    for intent in intents_list['intents']:
        if tag == intent['tag']:
            response = random.choice(intent['responses'])
            return response
    return "Lo siento, no entiendo tu pregunta."

def chatbot_response(msg):
    """Genera una respuesta del chatbot basada en el mensaje del usuario."""

    if "buscar" in msg.lower():
        query = msg.replace("buscar", "").strip()
        return buscar_en_internet(query)
    
    p = bow(msg, words, show_details=False)
    prediction = model.predict(np.array([p]))[0]
    max_index = np.argmax(prediction)
    tag = classes[max_index]
    return get_response(intents, tag)
