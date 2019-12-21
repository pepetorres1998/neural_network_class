import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import preprocessing

if(__name__ == '__main__'):
    x = pd.read_csv('./xbostonTest.csv')
    y = pd.read_csv('./ybostonTest.csv')
    y = y.values.ravel()
    lab_enc = preprocessing.LabelEncoder()
    y = lab_enc.fit_transform(y)

    x_train, x_test, y_train, y_test =\
        train_test_split(x,y,test_size=0.50,random_state=5)

    lin_model = LogisticRegression(max_iter=2000,
                        solver='lbfgs', verbose=10,
                        random_state=21,tol=0.000000001)

    lin_model.fit(x_train, y_train)
    print()

    y_train_predict = lin_model.predict(x_train)
    y_test_predict = lin_model.predict(x_test)

    df = pd.DataFrame({'Actual': y_train.flatten(), 'Predicted': y_train_predict.flatten()})
    df1 = df.head(25)
    df1.plot(kind='bar',figsize=(16,10))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()

    # model evaluation for train set
    rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
    r2 = r2_score(y_train, y_train_predict)

    print("El performance del modelo para el dataset de entrenamiento")
    print("-----------------------------------------------------------")
    print('Error cuadrado medio (RMSE) es: {}'.format(rmse))
    print('Coeficiente de determinación (r2_score) es {}'.format(r2))
    print("\n")

    # model evaluation for testing set
    rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
    r2 = r2_score(y_test, y_test_predict)

    print("El performance del modelo para el dataset de prueba")
    print("-----------------------------------------------------------")
    print('Error cuadrado medio (RMSE) es: {}'.format(rmse))
    print('Coeficiente de determinación (r2_score) es {}'.format(r2))
