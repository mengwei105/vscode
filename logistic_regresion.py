

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from sklearn.preprocessing import PolynomialFeatures

pd.set_option("display.notebook_repr_html", False)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 150)
pd.set_option("display.max_seq_items", None)

# %matplotlib inline

sns.set_context("notebook")
sns.set_style("white")


def loaddata(file, delimiter):
    data = np.loadtxt(file,delimiter=delimiter)
    print("Dimensions:", data.shape)
    print(data[1:6, :])
    return data

def plotData(data,label_x,label_y,label_pos,label_neg,axes=None):
    neg=data[:,2]==0
    pos=data[:,2]==1
    if axes==None:
        axes=plt.gca()
    axes.scatter(data[pos][:,0],data[pos][:,1],marker="+",c="k",s=60,linewidth=2,label=label_pos)
    axes.scatter(data[neg][:,0],data[neg][:,1],c="y",s=60,label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True,fancybox=True)
    # plt.show()

data=loaddata("data1.txt",',')

x=np.c_[np.ones((data.shape[0],1)),data[:,0:2]]
y=np.c_[data[:,2]]


plotData(data,"Exam 1 score","Exam 2 score","Pass","Fail")

def sigmoid(z):
    return(1/(1+np.exp(-z)))

def costFunction(theta,x,y):
    m=y.size
    h=sigmoid(x.dot(theta))

    J=-1*(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))

    if np.isnan(J[0]):
        return np.inf
    return J[0]

def gradient(theta,x,y):
    m=y.size
    h=sigmoid(x.dot(theta.reshape(-1,1)))

    grad=(1/m)*x.T.dot(h-y)

    return grad.flatten()

initial_theta=np.zeros(x.shape[1])
cost=costFunction(initial_theta,x,y)
grad=gradient(initial_theta,x,y)
print("Cost: \n",cost)
print("Grade:\n",grad)

res=minimize(costFunction,initial_theta,args=(x,y),method=None,jac=gradient,options={"maxiter":400})

def  predict(theta,x,threshold=0.5):
    p=sigmoid(x.dot(theta.T))>=threshold
    return p.astype("int")

sigmoid(np.array([1,45,85]).dot(res.x.T))

p=predict(res.x,x)
print("Train accuracy {}%".format(100*sum(p==y.ravel())/p.size))

plt.scatter(45,85,s=60,c="r",marker="v",label="(45,85)")
plotData(data,"Exam 1 score","Exam 2 core","pass","Fails")
x1_min,x1_max=x[:,1].min(),x[:,1].max(),
x2_min,x2_max=x[:,2].min(),x[:,2].max(),
xx1,xx2=np.meshgrid(np.linspace(x1_min,x1_max),np.linspace(x2_min,x2_max))
h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(res.x))
# h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(res.x))
h=h.reshape(xx1.shape)
plt.contour(xx1,xx2,h,[0.5],linewidth=1,colors="b")
plt.show()