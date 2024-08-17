import numpy as np
import json
import random  # Importar random
import pickle  # Importar pickle
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer

# Carga del modelo
model = load_model('chatbot_model.h5')

# Carga de los datos de intenciones y clases
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    # Tokeniza la oración y aplica lematización
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    # Crea una bolsa de palabras para la oración
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    # Predice la clase de la oración
    bag = bag_of_words(sentence)
    prediction = model.predict(np.array([bag]))[0]
    max_index = np.argmax(prediction)
    return classes[max_index]

def get_response(intents_json, predicted_class):
    # Obtiene la respuesta correspondiente a la intención predicha
    for intent in intents_json['intents']:
        if intent['tag'] == predicted_class:
            response = random.choice(intent['responses'])
            return response

print("Bot is running!")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    predicted_class = predict_class(user_input)
    response = get_response(intents, predicted_class)
    print("Bot:", response)
