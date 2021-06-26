import numpy as np
import cv2
import cvlib as cv

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

                    
model = load_model('detectarea_genului_pers3.h5')

# initializare camera
webcam = cv2.VideoCapture(0)
    
clase = ['Barbat','Femeie']

while webcam.isOpened():

    # citire frame (img/s) de la camera 
    _, frame = webcam.read()

    fata,_ = cv.detect_face(frame)
    
    if not fata:
        cv2.putText(frame, 'Momentan nu exista nimic ce poate fi detectat!', (12, 472), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
        
    for i, f in enumerate(fata):
         
        (start_X, start_Y) = f[0], f[1]
        (final_X, final_Y) = f[2], f[3]
   
        cv2.rectangle(frame, (start_X,start_Y), (final_X,final_Y), (0, 204, 102), 2)    # B G R
        
        fata_decupata = np.copy(frame[start_Y:final_Y,start_X:final_X])
        
        if (fata_decupata.shape[0]) < 10 or (fata_decupata.shape[1]) < 10:
            continue
       
        fata_decupata = cv2.resize(fata_decupata, (96,96))
        fata_decupata = fata_decupata.astype("float") / 255.0
        fata_decupata = img_to_array(fata_decupata)
        fata_decupata = np.expand_dims(fata_decupata, axis=0)

        val_incredere = model.predict(fata_decupata)[0] 
        
        i = np.argmax(val_incredere)
        eticheta = clase[i]

        eticheta = "{} - {:.1f}%".format(eticheta, val_incredere[i] * 100)

        Y = start_Y - 10 if start_Y - 10 > 10 else start_Y + 10

        cv2.putText(frame, eticheta, (start_X, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 204, 102), 2)

    cv2.imshow(" ", frame)

    # la apasarea tastei "x" se opreste executia programului
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# eliberare resurse
webcam.release()
cv2.destroyAllWindows()
