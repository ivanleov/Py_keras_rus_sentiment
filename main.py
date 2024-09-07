import os.path
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, MaxPooling1D, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import translators as ts
from tensorflow.python.keras.models import save_model

Form, Window = uic.loadUiType("1.ui")     #интерфейс
app = QApplication([])
window = Window()
form = Form()
form.setupUi(window)
window.show()
num_words = 3000       #колво отзывов
max_review_len = 50      #макс длина отзыва из базы

train = pd.read_csv('yelp_review_polarity_csv/train.csv',   # подключаем базу отзывов с сайта YELP для обучения и тренировки
                        header=None,
                        names=['Class', 'Review'])
tokenizer = Tokenizer(num_words=num_words)  #токенизируем слова в базе
reviews = train['Review']
tokenizer.fit_on_texts(reviews)

if os.path.exists('model.keras'):            #если модель нейронки существует подключай её
    model = keras.models.load_model('model.keras')

else:  #иначе создавай модель и обучай её


    y_train = train['Class'] - 1
    sequences = tokenizer.texts_to_sequences(reviews)
    index = 100
    x_train = pad_sequences(sequences, maxlen=max_review_len)
    model = Sequential()                                                 #модель секвеншл ограничена 3000 словами, максимальная длина отзыва - 50 слов
    model.add(Embedding(num_words, 64, input_length=max_review_len))
    model.add(Conv1D(250, 5, padding='valid', activation='relu'))        # 3 слоя нейронки (300,128, 1 нейрон)
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',          #компилируем модель
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train,                     #обучаем модель в 4 эпохи по трейн выборке
                        y_train,
                        epochs=4,
                        batch_size=128,
                        validation_split=0.1)
    model.save('model.keras')                        #сохраняем полученную модель

def getlabel():                          #бери написанный в окошке текст
    return form.textEdit.toPlainText()






def clickb1():       #по нажатии на кнопку прогоняй текст через нейронку
    text = ts.translate_text(str(getlabel()), from_language='ru', to_language='en')      #переводи текст на англ
    sequence = tokenizer.texts_to_sequences([text])          #токенизируй
    data = pad_sequences(sequence, maxlen=max_review_len)
    result = model.predict(data)                             #получай результат 0 или 1

    if result > 0.5:                                         # при рез. больше половины текст положительный, иначе наоборот
        text = "Результат оценки: положительный отзыв"
    else:
        text = "Результат оценки: отрицательный отзыв"
    form.label_3.setText(text)                               #помести результат  в окно


form.pushButton.clicked.connect(clickb1)  # метод по нажатии на кнопку






app.exec_()      #покидай интерфейс
