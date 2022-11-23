from torch.utils.data import Dataset
import h5py


class H5DData(Dataset):

    def __init__(self, archivo, transform=None):
        self.archivo = h5py.File(archivo, 'r')
        self.labels = self.archivo['Y_TrainSet']
        self.data = self.archivo['X_TrainSet']
        self.transform = transform

    def __getitem__(self, index):
        datum = self.data[index]
        if self.transform is not None:
            datum = self.transform(datum)
        return datum, self.labels[index]

    def __len__(self):
        return len(self.labels)

    def close(self):
        self.archivo.close()
