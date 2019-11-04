from sklearn import datasets
from sklearn import svm
import numpy as np 
import matplotlib.pyplot as plt 

#import irir data to model Svm classifier
iris_dataset = datasets.load_iris()
print("Iris data set Description :: ", iris_dataset['DESCR'])   
print("Iris feature data :: ", iris_dataset['data'])
print("Iris target :: ", iris_dataset['target'])

def visualize_sepal_data():
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target
    plt.scatter(X[:, 0],X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('Sepal Width & Length')
    plt.show()
visualize_sepal_data()
def visual_petal_data():
    iris = datasets.load_iris()
    X    = iris.data[:, :2]
    y    = iris.target
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Petal length')
    plt.ylabel('Petak width')
    plt.title('Petal Width & Length')
    plt.show()
visual_petal_data()
#Modeling Different Kernel Svm classifier using Iris Sepal features
"""iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
C = 1.0
svc = svm.SVC(kernel='linear', C = C).fit(X,y)
lin_svc = svm.LinearSVC((C=C).fit(X,y))
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X,y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X,y)
#Visualizing the modeled svm classifiers with Iris Sepal features
h = 0.2
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].min() + 1
xx, yy = np.meshgrid(np.arrange(x_min, x_max, h), np.arange(y_min, y_max, h))

titles = ['SVC with linear kernel','LinearsSVC(linear kernel)','SVC with RBF kernel','SVC with polynomial (degree 3) kernel']
for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    plt.subplot(2, 2, i + 1) 
    plt.subplot_adjust(wspace=0.4, hspace=0.4)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx,yy,Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(),xx,max())
    plt.ylim(yy.min(),yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()"""

    


