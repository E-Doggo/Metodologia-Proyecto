import h5py
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import sklearn.metrics as mt

var = h5py.File("./entrenamiento/DataSetEmpleados.h5")

X = var['X_TrainSet'][:]

y = var['Y_TrainSet'][:]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


clasificador = SVC(kernel = "sigmoid")
clasificador.fit(x_train, y_train)

y_pred = clasificador.predict(x_test)

accuracy = mt.accuracy_score(y_test,y_pred)*100
confusion_mat = mt.confusion_matrix(y_test,y_pred)
f1_score = mt.f1_score(y_test,y_pred, average = None)
recall = mt.recall_score(y_test,y_pred, average = None)

print(accuracy)
print(confusion_mat)
print(f1_score)
print(recall)
