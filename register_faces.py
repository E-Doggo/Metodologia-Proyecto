import matplotlib.pyplot as pl
import cv2
import os

count = 0
imagenID = str(input("Ingrese ID: ")).lower()
direccion = './imagenes/'+imagenID

def crearDirectorio():
    if(not os.path.exists(direccion)):
        os.makedirs(direccion)

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

def capturarcamara():
    video_capture = cv2.VideoCapture(0)
    crearDirectorio()
    count = 0
    while True:
        ret, frame = video_capture.read()
        gris, rostro = agrupa_rostro(frame)
        i = 0
        for face in rostro:
            (x, y, w, h) = face
            alto = int(h*1.6)
            p1 = int (y+h // 2) - alto // 2
            p2 = int (x+w // 2) - alto // 2
            cara = frame[p1: p1 + alto, p2: p2 + alto]
            #cara = numpy.resize(cara,(64,64))
            cv2.imwrite((os.path.join(direccion, str(count) + '.jpg')), cara)
            count = count + 1
            if w > 100:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
        cv2.imshow('Video', frame)
        
        if (cv2.waitKey(1) & 0xFF == ord('q')) or count>=250:
            break

    video_capture.release()
    cv2.destroyAllWindows()
    
capturarcamara()


