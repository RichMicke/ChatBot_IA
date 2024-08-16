import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

# Importación de las bibliotecas necesarias

# keras es una biblioteca para construir y entrenar modelos de redes neuronales
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import sgd_experimental

# Inicialización del lematizador de WordNet.
# El lematizador convierte palabras a su forma base o lema.
lemmatizer = WordNetLemmatizer()

# Carga del archivo "intents.json" que contiene las intenciones o categorías para el modelo de chatbot.
# Este archivo generalmente incluye patrones y respuestas para diferentes intenciones.
intents = json.loads(open("intents.json").read())

# Descarga de los paquetes necesarios de NLTK (Natural Language Toolkit):
# 'punkt': Tokenizador para dividir el texto en palabras o frases.
# 'wordnet': Base de datos de palabras en inglés, utilizada para el lematizador.
# 'omw-1.4': Paquete de WordNet Open Multilingual Wordnet.
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']

# Itera sobre cada intención en el archivo intents.json
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lematiza las palabras y elimina los caracteres ignorados
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Guarda las palabras y clases en archivos pickle
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

training = []
output_empty = [0] * len(classes)

# Creación del conjunto de entrenamiento
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    oupout_row = list(output_empty)
    oupout_row[classes.index(document[1])] = 1
    training.append([bag,oupout_row])

random.shuffle(training)
training = np.array(training)

print(training)
    

