'''
University at Buffalo - Spring 2015
CSE 574 - Introduction to Machine Learning
Programming Assignment 2

Authors: Dheeraj Balakavi, Pravin Umamaheswaran, Mithun

'''
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import pickle

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    
    
    means = np.array([])
    unique_y = np.unique(y)
    
    i = 0;
    # processing each unique values in batches
    for index_val in unique_y:
        
        mapp_matrix = (y == index_val)
        mapp_matrix.shape = ((1,X.shape[0])) 
        #extended matrix
        ex_mapp_matrix = np.tile(mapp_matrix,X.shape[1])
        ex_mapp_matrix = np.reshape(ex_mapp_matrix,(X.shape[1],X.shape[0]))
        ex_mapp_matrix = ex_mapp_matrix.T
       
        MeanMat = np.asarray(X[ex_mapp_matrix])
        new_row = MeanMat.shape[0]/X.shape[1]
       
        KIK = np.reshape(MeanMat,(new_row,X.shape[1]))
        #taking the maen values of i/p per input data variable(x^i) per class
        Mean_Val = KIK.mean(0)
   
        if i == 0:
            means = np.hstack(Mean_Val)
        else :
            means = np.vstack((means,Mean_Val))  

        i+=1  
          
    means = means . T
    covmat = np.cov(X.T) # i am not sure about the covarience
        
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    covmats = list()
    unique_classes = np.unique(y)
    mean_vectors = []
    for cl in range(1,5):
        mean_vectors.append(np.mean(X[y==cl], axis=0))
    
    i =0 
    for class_val in unique_classes:
         mapping_matrix = (y==class_val)
         mapping_matrix.shape = ((1,X.shape[0]))
         
         # creating an extended matrix by using tile function
         # to match the dimensions and separate by class
            
         ext_map_matrix = np.tile(mapping_matrix,X.shape[1])
         ext_map_matrix = np.reshape(ext_map_matrix,(X.shape[1],X.shape[0]))
         ext_map_matrix = ext_map_matrix.T
         
         class_ip = np.asarray(X[ext_map_matrix])
         row_size = class_ip.shape[0]/X.shape[1]
         
         class_ip = np.reshape(class_ip,(row_size,X.shape[1]))
         covmat = np.cov(class_ip.T)
         covmats.append(covmat)
       
         #take mean values of i/p per input data variable per class
         mean_value = class_ip.mean(0)
         if i == 0:
            means = np.hstack(mean_value)
         else :
            means = np.vstack((means,mean_value))  

         i+=1  
          
    means = means . T
  
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    no_of_classes = 5
    inv_covariance = np.linalg.inv(covmat)  
    N = Xtest.shape[0]
   
    calc_label = np.array([])
    #calc_label = np.reshape(calc_label,(100,1))
    new_mean = means.T
    i = 0
    for each_row in range(N):   # for each row data calculate the probability or the classes
         row_data = Xtest[each_row,:]
         pdf_vector = np.array([])
         for each_class in range(no_of_classes):
             X_MU = row_data - new_mean[each_class]
             intmdt = np.dot(X_MU.T,inv_covariance)
             temp    = np.dot(intmdt,X_MU)
             d = 1
             pdf = np.exp(-1/2*temp)
             pdf_vector = np.append(pdf_vector,pdf)
             #pdf_vector[each_class] = pdf
         calc_label = np.append(calc_label,[np.argmax(pdf_vector)+1]) 
    calc_label  = np.matrix(calc_label).T      
   
    acc = 100 * np.mean((calc_label == ytest).astype(float)) 
    #plotGraph(Xtest,ytest,calc_label)
    return acc

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    #no_of_classes = 5
    k = 5
    N = Xtest.shape[0]
    i= 0
    new_mean = means.T
    calc_label = np.array([])
    for each_row in range(N):  # for each row of the data calculate the probability 
      x_i = Xtest[each_row,:]
      pdf_vector = np.array([])
      for cl,cov_mat in zip(range(1,6),covmats):
          inv_covmat = np.linalg.inv(cov_mat) # inverse covariance matrix
          normlz_factor = 1/np.sqrt((2*np.pi)**k * np.linalg.det(cov_mat))
         
          X_MU   = x_i - new_mean[cl-1]
          intmdt = np.dot(X_MU.T,inv_covmat)
          temp   = np.dot(intmdt,X_MU)
          pdf    = np.exp(-1/2*temp)*normlz_factor
          pdf_vector = np.append(pdf_vector,pdf)
      calc_label = np.append(calc_label,[np.argmax(pdf_vector)+1])
    
    calc_label = np.matrix(calc_label).T      
    
    acc = 100 * np.mean((calc_label == ytest).astype(float)) 
    
    return acc

# Author: Dheeraj Balakavi
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1
def learnOLERegression(X,y):
    # Step-1: Compute the Pseudo-Inverse of X --> PI(X)
    XT = X.transpose()
    XT_into_X = np.dot (XT, X)
    XT_into_X_inv = np.linalg.inv(XT_into_X)
    
    X_pi = np.dot (XT_into_X_inv, XT)
    
    # Test to see if I got it right.. if we do (X-pi * x) we must get an identity matrix 
    # as it is pseudo inverse we are doing
    X_ident = np.dot (X_pi, X)
    
    # Step-2: Compute w
    w = np.dot (X_pi, y)                           
    return w

# Author: Dheeraj Balakavi
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1   
def learnRidgeRegression(X,y,lambd):
    X_rows, X_columns = X.shape
    N = X_rows
    M = X_columns
    # Step-1: Compute the Pseudo-Inverse of X with regularization co-eff
    XT = X.transpose()
    XT_into_X = np.dot (XT, X)
    identity = np.identity (M)
    lambd_into_I = (lambd * N )* identity
    
    sum_lambI_XTX = lambd_into_I + XT_into_X
    sum_lambI_XTX_inv = np.linalg.inv(sum_lambI_XTX)
    
    # Step-2: Compute w
    product_one = np.dot (sum_lambI_XTX_inv, XT)
    w = np.dot (product_one, y)
                                                
    return w

# Author: Dheeraj Balakavi
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
def testOLERegression(w,Xtest,ytest):
    # Step-1: Compute the root mean square error (rmse)
    X_rows, X_columns = Xtest.shape
    N = X_rows
    
    wT = w.transpose()
    wT_into_Xtest = np.dot (Xtest, wT.transpose())
    
    subVal = np.subtract (ytest, wT_into_Xtest)
    subVal_sqr = subVal * subVal
    
    sumVal = np.sum (subVal_sqr)
    
    sumVal_sqrt = np.sqrt(sumVal)
    rmse = sumVal_sqrt / N

    return rmse

# Author: Dheeraj Balakavi
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda 
def regressionObjVal(w, X, y, lambd):
    
    # Step-1: Error computation --> scalar
    X_rows, X_columns = X.shape
    N = X_rows
    w = np.vstack (w)
    
    Xw = np.dot (X, w)
    y_minus_Xw = y - Xw
    y_minus_Xw_transpose = y_minus_Xw.transpose()
     
    prod1 = (np.dot (y_minus_Xw_transpose, y_minus_Xw) / (2 * N) )    
    prod2 = (1/2) * lambd * np.dot (w.transpose(), w)    
    error = (prod1 + prod2).item(0)
    
    # Step-2: Error grad computation --> vector
    XT_X = np.dot (X.transpose(), X)
    wT_XT_X = np.dot (w.transpose(), XT_X)  
    yT_X = np.dot (y.transpose(), X)
    
    sum1 = (wT_XT_X - yT_X) / N
    sum2 = np.dot (w.transpose(), lambd)
    error_grad = sum1 + sum2
    
    error_grad = np.squeeze(np.asarray(error_grad))                            
    return error, error_grad


def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    output = []
    for val in np.nditer(x):
        
        #if p == 0:
        #    Xd = np.ones((x.shape[0],1))
        #    break
        
        for power  in range(0,p+1):
            #print 'awesome'            
            output.append(pow(val,power))

    #print 'x:' + str(x.shape[0])
    #print 'p:' + str(p)
    #print 'Size is:' + str(len(output))
    Xd = np.asarray(output)
    Xd = np.reshape(Xd,(x.shape[0],p+1))
    return Xd

# Main script

# Problem 1
# load the sample data 
                                                               
X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))            

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# Problem 2
X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))   
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))


# Problem 3
k = 21
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)

# Problem 4
k = 21
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l_1 = np.zeros((X_i.shape[1],1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    rmses4[i] = testOLERegression(w_l_1,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)

# Problem 5
 
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
 
