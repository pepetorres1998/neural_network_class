import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

if(__name__ == '__main__'):
    x = pd.read_csv('./xsonarTest.csv')
    y = pd.read_csv('./ysonarTest.csv')
    y = y.values.ravel()

    x_train, x_test, y_train, y_test =\
        train_test_split(x,y,test_size=0.30,random_state=27)

    clf = LogisticRegression(max_iter=5000,
                        solver='lbfgs', verbose=10,
                        random_state=21,tol=0.000000001)

    clf.fit(x_train, y_train)
    print()

    y_pred = clf.predict(x_test)

    print("Accuracy score:")
    print(accuracy_score(y_test, y_pred))
    print()

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)

    sns.heatmap(cm, center=True)
    plt.show()
