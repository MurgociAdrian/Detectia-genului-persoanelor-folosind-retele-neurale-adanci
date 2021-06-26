import numpy as np
import cv2
import cvlib as cv

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

                    
model = load_model('detectarea_genului_pers3.h5')
 
clase = ['Barbat','Femeie']

# citire imagine
img_in = cv2.imread(r'D:/FACULTATE/LICENTA/fata2.jpg') # dim (y,x,3)

dim1 = img_in.shape[0]  # y
dim2 = img_in.shape[1]  # x
aux1 = 1
aux2 = 1
aux_font = 1
dim_font_1 = 1.7
dim_font_2 = 0.6
val_grosime_font = 2

# in cazul in care imaginea de la intrare are dimensiuni mai mari decat 1920x1010 se va face redimensionare pentru afisare:
if (dim1 > 1010):  
    aux1 = 0
    dim1 = 1010
if (dim2 > 1920):
    aux1 = 0
    aux_font = 0
    dim2 = 1920

# conditie pentru imagini cu rezolutii foarte mici pentru afisarea textului in limitele acesteia    
if (dim2 < 900): 
    dim_font_1 = 0.7
    
# conditie pentru imagini cu rezolutii foarte mari: se cresc valorile care tin de dim font pentru ca textul sa fie vizibil
if (aux_font == 0): 
    dim_font_2 = 1
    val_grosime_font = 3

# functia pentru detectarea fetei: fata - o lista ce contine cele 4 puncte: 2 din coltul stanga jos(X,Y) respectiv 2 din dreapta sus(X,Y)    
fata,_ = cv.detect_face(img_in) 

# in cazul in care nu s-a gasit nicio fata se fac afisarile corespunzatoare:    
if not fata:
    if aux1 == 0:
        aux2 = 0
        img_redim = cv2.resize(img_in, (dim2,dim1))
        cv2.putText(img_redim, 'Nu s-a putut detecta nicio fata!', (10, (dim1-15)), cv2.FONT_HERSHEY_DUPLEX, dim_font_1, (0, 0, 255), 2)
    else:
        cv2.putText(img_in, 'Nu s-a putut detecta nicio fata!', (10, (dim1-15)), cv2.FONT_HERSHEY_DUPLEX, dim_font_1, (0, 0, 255), 2)

for i, f in enumerate(fata):
     
    # cele 4 puncte 
    (start_X, start_Y) = f[0], f[1]
    (final_X, final_Y) = f[2], f[3]
    
    # desenare dreptunghi peste fata detectata
    cv2.rectangle(img_in, (start_X,start_Y), (final_X,final_Y), (0, 204, 102), 2)   

    # decupare fata detectata
    fata_decupata = np.copy(img_in[start_Y:final_Y,start_X:final_X])
    
    if (fata_decupata.shape[0]) < 10 or (fata_decupata.shape[1]) < 10:
        continue
   
    # preprocesare: (la fel ca la construirea modelului)
    fata_decupata = cv2.resize(fata_decupata, (96,96))
    fata_decupata = fata_decupata.astype("float") / 255.0
    fata_decupata = img_to_array(fata_decupata)
    fata_decupata = np.expand_dims(fata_decupata, axis=0)

    # aplicarea modelului pentru fata curenta
    val_incredere = model.predict(fata_decupata)[0] 
      
    # se determina clasa corecta in functie de valoarea maxima de mai sus
    i = np.argmax(val_incredere)
    eticheta = clase[i]

    eticheta = "{} - {:.1f}%".format(eticheta, val_incredere[i] * 100)

    Y = start_Y - 10 if start_Y - 10 > 10 else start_Y + 10

    # functia de scriere a clasei si a valorii in procent de incredere a fetei detectate
    cv2.putText(img_in, eticheta, (start_X, Y), cv2.FONT_HERSHEY_SIMPLEX, dim_font_2, (0, 204, 102), val_grosime_font)


# afisare  
if ((aux2 == 0) & (aux1 == 0)):
    cv2.imshow(" ", img_redim)
if ((aux1 == 0) & (aux2 == 1)):
    img_redim = cv2.resize(img_in, (dim2,dim1))
    cv2.imshow(" ", img_redim)
if (aux1+aux2) == 2:
    cv2.imshow(" ", img_in)

