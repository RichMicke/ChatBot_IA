import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

# Inicialización del lematizador
lemmatizer = WordNetLemmatizer()

# Carga del archivo intents.json
with open("intents.json") as file:
    intents = json.load(file)

# Descarga de los paquetes necesarios de NLTK
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
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
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
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Verificar la consistencia de los datos antes de convertir a un array de NumPy
print(f"Longitudes de bag: {[len(item[0]) for item in training]}")
print(f"Longitudes de output_row: {[len(item[1]) for item in training]}")

# Dividir los datos en entrada (X) y salida (y)
X_train = np.array([item[0] for item in training])
y_train = np.array([item[1] for item in training])

# Define el modelo
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

# Compila el modelo
model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01, momentum=0.9), metrics=['accuracy'])

# Entrena el modelo
model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

# Guarda el modelo
model.save('chatbot_model.h5')
print("Modelo guardado como 'chatbot_model.h5'")
