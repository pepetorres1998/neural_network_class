import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Lee el dataset, que esta en la subcarpeta input
x = pd.read_csv('./xsonarTest.csv')
y = pd.read_csv('./ysonarTest.csv')
y = y.values.ravel()

# Define los conjuntos que se utilizaran para prueba
# y para entrenamiento, el procentaje
# que se utilizara para pruebas es el 30%
x_train, x_test, y_train, y_test =\
    train_test_split(x,y,test_size=0.30,random_state=27)

# Instancia la clase "MLPClassifier"
clf = MLPClassifier(hidden_layer_sizes=(100,100,100),
                    max_iter=3000, alpha=0.0001,
                    solver='lbfgs', verbose=10,
                    random_state=21,tol=0.000000001)

# Entrena la red neuronal
clf.fit(x_train, y_train)
print()

# Predice con la red neuronal y el conjunto de prueba
# x para poder determinar la exactitud de la
# red neuronal
y_pred = clf.predict(x_test)

print("Accuracy score:")
acc = accuracy_score(y_test, y_pred)
print(acc)
print()

print("Error: {}".format(1 - acc))
print()

# Genera una matriz de confusión para probar la red
# neuronal en acción. Para esto utiliza el conjunto
# de pruebas
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# Imprime el resultado de las mismas con un
# heatmap de  la librería seaborn
sns.heatmap(cm, center=True)
plt.show()
