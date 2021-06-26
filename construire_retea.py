import numpy as np
import os
import glob
import cv2
import random
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split


# parametrii initiali
epochs = 50
batch_size = 64
l_rate = 0.001
dim_img = (96,96,3)

# cele 2 liste unde o sa stochez imaginile/etichetele din baza de date
data_img = []
etichete = []

# incarcare imagini din baza de date locala
fisiere_img = [f for f in glob.glob(r'C:\p_licenta\bazadetest' + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(fisiere_img)

# convertire imagini in matrici si redimensionare + etichetare categorii si apoi salvare in liste
for c_img in fisiere_img:
    
    img = cv2.imread(c_img) # se converteste imaginea in numpy array  (B G R)
    img = cv2.resize(img, (dim_img[0],dim_img[1])) # redimensionare 96x96
    img = img_to_array(img)
    data_img.append(img)  

    aux_etich = c_img.split(os.path.sep)[-2] 
    if aux_etich == "femeie":
        aux_etich = 1
    else:
        aux_etich = 0
        
    etichete.append([aux_etich]) # [[1], [0], [0], ...]

# preprocesare:  convertesc listele declarate mai sus in matrici)
data_img = np.array(data_img, dtype="float") / 255.0
etichete = np.array(etichete)

# divizare baza de date astfel:  20% - img de validare (test) si 80% - img de antrenare (train)
(train_X, test_X, train_Y, test_Y) = train_test_split(data_img, etichete, test_size=0.2, random_state=42)

# convertirea matricilor de etichete in matrici binare folosind conventia:
train_Y = to_categorical(train_Y, num_classes=2) # 0 - barbat va fi [1, 0],  1 - femeie va fi [0, 1]
test_Y = to_categorical(test_Y, num_classes=2)  # [[1, 0], [0, 1], [0, 1], ...] 

# augmentare
aug = ImageDataGenerator(rotation_range=25)

# definire model
def build(latime, inaltime, nr_canale_culori, clase):   # 96  96  3  2
    model = Sequential()
    inputShape = (latime, inaltime, nr_canale_culori)
    dim_canal = -1

    if K.image_data_format() == "channels_first": # returneaza un string, fie 'channels_first' sau 'channels_last'
        inputShape = (nr_canale_culori, inaltime, latime)
        dim_canal = 1 # utilizat la normalizare, iar channel se refera la nr_canale_culori
    
    # Bloc 1
    model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=dim_canal))
    model.add(MaxPooling2D(pool_size=(3,3)))    
    model.add(Dropout(0.25))    

    # Bloc 2
    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=dim_canal))

    # Bloc 3
    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=dim_canal))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # Bloc 4
    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=dim_canal))

    # Bloc 5
    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=dim_canal))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # Bloc 6
    model.add(Flatten())
    
    model.add(Dense(1024)) 
    model.add(Activation("relu")) 
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Bloc 7
    model.add(Dense(clase))
    model.add(Activation("sigmoid")) 

    return model


# construire model
model = build(latime=dim_img[0], inaltime=dim_img[1], nr_canale_culori=dim_img[2], clase=2)

# print(model.summary())

# compilare model
opt = Adam(learning_rate=l_rate, decay=l_rate/epochs) # functia de optimizare
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# antrenare model
H = model.fit(aug.flow(train_X, train_Y, batch_size=batch_size),
                        validation_data=(test_X,test_Y),
                        steps_per_epoch=len(train_X) // batch_size, 
                        epochs=epochs, verbose=1)    

# salvare model 
model.save('detectarea_genului_pers3.h5')


# grafice pentru training si validation accuracy+loss in functie de epochs
plt.style.use("ggplot")

plt.figure()
plt.plot(np.arange(0,epochs), H.history['accuracy'], label="train_acc", color='k')
plt.plot(np.arange(0,epochs), H.history['val_accuracy'], label="valid_acc", color='b')
plt.title("Training/Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig('grafic31.png')

plt.figure()
plt.plot(np.arange(0,epochs), H.history['loss'], label="train_loss", color='y')
plt.plot(np.arange(0,epochs), H.history['val_loss'], label="valid_loss", color='r')
plt.title("Training/Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig('grafic32.png')



