#Logistic Regression
#from maths to coding
import numpy as np 

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

#to chuc du lieu
X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
            2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T 
y = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]]).T 
#them so 1 vao moi diem du lieu
one = np.ones((X.shape[0], 1))
X = np.concatenate((X, one), axis = 1)


#theta_init = np.random.rand(1, X.shape[1])[0]
#theta, it = gradient_descent(X, y, theta_init) 
#print(f'Thete = {theta}, it = {it}')  
#hours = float(input("Nhap vao so gio: "))
#predict = 1 / (1 + (np.exp( -1 * (theta[0] * hours + theta[1]))))
#print("Ty le dau la = ", predict)

theta_init = np.random.rand(1, X.shape[1])[0]
theta, it  = gradient_descent(X, y, theta_init)
print("Theta= ",theta, ", Iteration= ", it)
np.savetxt('theta.txt', theta)
