import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

if(__name__=='__main__'):
    print(check_output(["ls", "./input"]).decode("utf8"))
    df = pd.read_csv('./input/Dataset_spine.csv')

    df = df.drop(['Unnamed: 13'], axis=1)

    print(df.head())
    print()

    print(df.describe())
    print()

    y = df['Class_att']
    x = df.drop(['Class_att'], axis=1)

    x_train, x_test, y_train, y_test =\
        train_test_split(x,y,test_size=0.30,random_state=27)

    clf = MLPClassifier(hidden_layer_sizes=(100,100,100),
                        max_iter=500, alpha=0.0001,
                        solver='sgd', verbose=10,
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
