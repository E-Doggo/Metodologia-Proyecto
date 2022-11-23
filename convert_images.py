import os
from PIL import Image
from sklearn import preprocessing
import numpy
import h5py


def obtener_imagenes(direccion):
    imagenes = []
    for file in os.listdir(direccion):
        name =  os.path.join(direccion, file)
        img = Image.open(name)
        img = img.resize((128,128), Image.ANTIALIAS)
        img = numpy.array(img)
        imagenes.append(img)
    arrayImagenes = numpy.array(imagenes)
    imagenes.clear()
    return arrayImagenes
        
def obtener_etiquetas(direccion):
    clases = []
    if (len(os.listdir(direccion))!= 0):
        for directory in os.listdir(direccion):
            clases.append(directory)
        label = preprocessing.LabelEncoder().fit_transform(clases)
        return clases, label
    
    
        
#separar clasificador de labels y clasificador de imagenes en dos funciones
lista = []
for directory in os.listdir('imagenes/'):
    lista.append(obtener_imagenes('imagenes/' + directory))
lista = numpy.array(lista)



labels = obtener_etiquetas('imagenes/')
print(labels)
y = numpy.repeat(labels[1], 250)
print(y)

lista1 = lista.reshape(len(lista)*250, 128, 128, 3)


directorio= 'entrenamiento/DataSetEmpleadosPyTorch.h5'

h5 = h5py.File(directorio, 'w')
h5.create_dataset('Y_TrainSet', data = y)
h5.create_dataset('X_TrainSet', data = lista1)
h5.close()

lista2 = lista.reshape(len(lista)*250, 49152)


directorio= 'entrenamiento/DataSetEmpleados.h5'

h5 = h5py.File(directorio, 'w')
h5.create_dataset('Y_TrainSet', data = y)
h5.create_dataset('X_TrainSet', data = lista2)
h5.close()




