import cv2
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as rna
import os
from datetime import datetime
import json

personas =[]

for directory in os.listdir('./imagenes/'):
    personas.append(directory)

today = datetime.now()
fecha = today.strftime("%Y-%m-%d")
hora = today.strftime("%H:%M:%S")

def saveToJson(persona):
    dictionary = {
    "nombre": persona,
    "fecha": fecha,
    "hora": hora,
    }
    return dictionary

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
model_ft = torch.load("./entrenamiento/modelo_caras.pt")



def agrupa_rostro(frame):
    haar_xml = "haarcascade_frontalface_default.xml"
    modelo = cv2.CascadeClassifier(cv2.data.haarcascades + haar_xml)
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    rostros = modelo.detectMultiScale(
            gris,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    return gris, rostros

camera = cv2.VideoCapture(0)



while True:
    ret, frame = camera.read()
    gris, rostro = agrupa_rostro(frame)
    i = 0
    for face in rostro:
        (x, y, w, h) = face
        alto = int(h*1.6)
        p1 = int (y+h // 2) - alto // 2
        p2 = int (x+w // 2) - alto // 2
        cara = frame[p1: p1 + alto, p2: p2 + alto]
        if cara.shape[0] > 20 and cara.shape[1] > 20:

            cara = cv2.resize(cara, (128, 128), interpolation=cv2.INTER_AREA)
            cara = cv2.dilate(cara, (3, 3,))

            cara = transform(cara)
            cara.unsqueeze_(dim=0)
            cara = Variable(cara)

            cara = cara.view(cara.shape[0], -1)
            predict = rna.softmax(model_ft(cara), dim=1)
            
            if w>100:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(personas[predict.argmax().item()]), (face[0], face[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 255, 0))

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        diccionario = saveToJson(str(personas[predict.argmax().item()]))
        with open("./verificacion/sample.json", "a+") as outfile:
            json.dump(diccionario, outfile)
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
