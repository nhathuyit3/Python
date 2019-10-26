#train model file
import numpy as np 
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('mnist-original', data_home='./')
N,d = mnist.data.shape
print(N)
print(d)
x_all = mnist.data 
y_all = mnist.target
#in thu 1 so trong datasets
import matplotlib
import matplotlib.pyplot as plt
plt.imshow(x_all.T[:,3000].reshape(28,28))
plt.axis("off")
plt.show()
#loc lai chi con 2 so 0 va 1
x0 = x_all[np.where(y_all == 0)[0]]
x1 = x_all[np.where(y_all == 1)[0]]
y0 = np.zeros(x0.shape[0])
y1 = np.ones(x1.shape[0])
#gop 0 va 1 lai thanh 1 tap dataset
x = np.concatenate((x0,x1), axis = 0)
y = np.concatenate((y0, y1))

one = np.ones((x.shape[0], 1))
X = np.concatenate((x, one), axis = 1)
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score
#chia tap su lieu thanh 2 tap train va set 
#x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size = 1000)
#train model
#model = LogisticRegression(C = 1e5)
#model.fit(x_train, y_train)
#check accuracy
#y_prediction = model.predict(x_test)
#print("Accuracy:" + str(100 * accuracy_score(y_test, y_prediction)))
#save model to file
#from sklearn.externals import joblib
#joblib.dump(model, "digits.pkl", compress = 3)
#mo file Handdigit v1 ra va modifycode

#trong doan code for rects in rects bo sung them doan code de lay vung anh 1 dc ve mang 28 x 28
def gradient_descent(X,y, theta_init, eta= 0.05):
    theta_old = theta_init
    theta_epoch = theta_init
    N = X.shape[0]
    for it in range(100_000):
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[i, :]
            yi = y[i]
            hi = 1 / (1 + np.exp(-np.dot(xi, theta_old.T)))
            gi = (yi - hi) * xi
            theta_new = theta_old + eta * gi
            theta_old = theta_new
        if np.linalg.norm(theta_epoch - theta_old) < 1e-4:
            break
        #trong vong for lon
        theta_epoch = theta_old
    return theta_epoch, it

#print(X.shape[1])
theta_init = np.random.rand(1, X.shape[1])[0]
theta, it  = gradient_descent(X, y, theta_init)
print("Theta= ",theta, ", Iteration= ", it)
print(theta.shape)
np.savetxt('theta.txt', theta)



