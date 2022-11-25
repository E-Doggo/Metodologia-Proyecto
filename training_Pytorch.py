import torch
from time import time
from torchvision import transforms
from torch import nn, optim
from H5DDATA import H5DData
import os

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_loader = torch.utils.data.DataLoader(H5DData("./entrenamiento/DataSetEmpleadosPyTorch.h5", transform), batch_size=64, shuffle=True)

capa_entrada = 128*128*3
capas_ocultas = [1024, 128, 64]
capa_salida = len(os.listdir('./imagenes'))


modelo = nn.Sequential(nn.Linear(capa_entrada, capas_ocultas[0]), nn.SELU(),
                       nn.Linear(capas_ocultas[0], capas_ocultas[1]), nn.SELU(),
                       nn.Linear(capas_ocultas[1], capa_salida), nn.LogSoftmax(dim=1))

j = nn.CrossEntropyLoss()

optimizador = optim.Adam(modelo.parameters(), lr=0.003)
tiempo = time()
epochs = 1
for e in range(epochs):
    costo = 0
    for imagen, etiqueta in train_loader:
        imagen = imagen.view(imagen.shape[0], -1)
        optimizador.zero_grad()
        h = modelo(imagen.float())
        error = j(h, etiqueta.long())
        error.backward()
        optimizador.step()
        costo += error.item()
    else:
        print("Epoch {} - Funcion costo: {}".format(e, costo / len(train_loader)))
print("\nTiempo de entrenamiento (en minutes) =", (time() - tiempo) / 60)

torch.save(modelo, './entrenamiento/modelo_caras.pt')

print(modelo)
