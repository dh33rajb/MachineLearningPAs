import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import matplotlib.pyplot as plt


def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
# Function written by Dheeraj    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    return  (1 / (1 + (np.exp(-z))))
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    
    #Pick a reasonable size for validation data
    
    print mat.keys()
    #Your code here
    train_data = np.array([])
    train_label = np.array([])
    validation_data = np.array([])
    validation_label = np.array([])
    test_data = np.array([])
    test_label = np.array([])
    
    # Dheeraj changes-1 start
    testLabelList = []
    testDataList = []
    trainLabelList = []
    trainDataList = []
    validationLabelList = []
    validationDataList = []   
    
    count =0;
    for key, value in mat.iteritems():
        #print key, mat[key]
        valDataCount = 0
        if "test" in key:
            #count = count + len(value)
            #np.vstack(
            for val in value:
                testLabelList.append(key)
                testDataList.append(val)
        elif "train" in key:
            for val in value:
                if valDataCount < 1000:
                    validationLabelList.append(key)
                    validationDataList.append(val)
                    valDataCount = valDataCount + 1
                else:
                    trainLabelList.append(key)
                    trainDataList.append(val)
    #print count
    #Dheeraj changes-1 end

    train_data = np.vstack(trainDataList)
    train_label = np.vstack(trainLabelList)
    validation_data = np.vstack(validationDataList)
    validation_label = np.vstack(validationLabelList)
    test_data = np.vstack(testDataList)
    test_label = np.vstack(testLabelList)
    
    #Type Conversion
    
    train_data = train_data.astype(float)
    validation_data = validation_data.astype(float)
    test_data = test_data.astype(float)
    
    #Normalization
    train_data = train_data/255
    validation_data = validation_data/255
    test_data = test_data/255
    
    #dheeraj changes-2 start
    """print len (train_label), len (train_label.T)
    print len(train_data), len (train_data.T)
    print len (validation_label), len (validation_label.T)
    print len(validation_data), len (validation_data.T)
    print len (test_label), len (test_label.T)
    print len(test_data), len (test_data.T)"""
    
    # trying to unflatten the row and plot the image of a number
    """fig = plt.figure(figsize=(12,12))
    row = train_data[0]
    print row
    plt.imshow(np.reshape(row,((28,28))))
    plt.axis('off')"""
    
    # dheeraj changes-2 end
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    
    
# Function written by Dheeraj  
def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    
    #Your code here
    #Dheeraj: Step-1: Feedforward Propogation starts here
    training_data_rows, training_data_cols = training_data.shape
    #training_data_cols = training_data_cols + 1
    train_data_with_bias = np.array([])
    train_data_with_bias_list = []
    for x in xrange (0, training_data_rows):
        train_data_with_bias_list.append(np.append ([1], training_data[x]))
    train_data_with_bias = np.vstack(train_data_with_bias_list)

    w1_transpose = w1.transpose()  
    a_all_data = np.dot (train_data_with_bias, w1_transpose)
    z_all_data = sigmoid (a_all_data)
    
    z_all_data_rows, z_all_data_columns = z_all_data.shape
    
    zj_all_data_with_bias = np.array([])
    zj_all_data_with_bias_list = []    
    for x in xrange (0, z_all_data_rows):
        zj_all_data_with_bias_list.append(np.append ([1], z_all_data[x]))
    zj_all_data_with_bias = np.vstack(zj_all_data_with_bias_list)
    
    w2_transpose = w2.transpose()
    b_all_data = np.dot (zj_all_data_with_bias,w2_transpose)
    o_all_data = sigmoid (b_all_data)
    #Dheeraj: Step-1: Feedforward Propogation ends here
    
    #Dheeraj: Step-2: error function starts here
    # Step-2.1: y_all_data computation
    y_temp = np.array([])
    y_Temp_list = []
    y_all_data = np.array([])
    y_all_data_list = []   
    for i in range (0, training_data_rows):
        y_Temp_list.append ([0, 0 ,0, 0, 0, 0, 0, 0, 0] )
    y_temp = np.vstack (y_Temp_list)
    for i in range (0, training_data_rows):
        target_value = int (training_label[i][0][-1:])
        y_all_data_list.append(np.insert (y_temp[i], target_value, 1))        
    y_all_data = np.vstack (y_all_data_list)
    # Step-2.1: y_all_data computation ends
    
    #Step-2.2: (1-y_all_data) and (1-o_all_data) computation starts here
    one_minus_y_all_data = 1 - y_all_data
    one_minus_o_all_data = 1 - o_all_data
    #Step-2.2: (1-y_all_data) and (1-o_all_data) computationends here
    
    #Step-2.3: ln (o_all_data) and ln (1- o_all_data) computation starts
    ln_o_all_data = np.log (o_all_data)
    ln_one_minus_o_all_data = np.log (one_minus_o_all_data)
    #Step-2.3: ln (o_all_data) and ln (1- o_all_data) computation ends
    
    #Step-2.4 computing j_all_data and j_error_function starts here
    """ln_o_all_data_transpose = ln_o_all_data.transpose()
    ln_one_minus_o_all_data_transpose = ln_one_minus_o_all_data.transpose()
    temp_1 = np.dot (y_all_data, ln_o_all_data_transpose)
    temp_2 = np.dot (one_minus_y_all_data, ln_one_minus_o_all_data_transpose)
    j_all_data = np.add (temp_1, temp_2)"""
    
    j_all_data = np.array([])
    j_all_data_list = []
    j_error_function = 0
    for x in xrange (0, training_data_rows):
        j = np.add (np.dot (y_all_data[x], (ln_o_all_data[x]).transpose()), np.dot (one_minus_y_all_data[x], (ln_one_minus_o_all_data[x]).transpose()))
        j_all_data_list.append (j)
        j_error_function = j_error_function + j
    j_all_data = np.vstack (j_all_data_list)
    j_error_function = j_error_function / training_data_rows
    #Step-2.4 computing j_all_data ends here
    
    obj_val = j_error_function
    #Dheeraj: Step-2: error function ends here
    
    #Dheeraj: Step-3: back propogation section starts here
    
    
    
    #Dheeraj: Step-3: back propogation section ends here

    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    
    obj_grad = np.array([])
    return (obj_val,obj_grad)

def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    labels = np.array([])
    #Your code here
    
    return labels
    



"""**************Neural Network Script Starts here********************************"""

# Dheeraj: preprocess reads the mnist_all data file and spits out the test, train and validation numy arrays
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
# dheeraj: this extracts the column size of the training data numpy array
n_input = train_data.shape[1]; 
#print "------", n_input, "------"

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
# dheeraj: this is used to generate the weight matrices of random weights for the hidden and output layers
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
#dheeraj: here we flatten the above two weight vectors and form a single row vector out of them.
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Dheeraj: Code Testing Scripts
#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print '\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%'

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print '\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%'


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print '\n Test set Accuracy:' + + str(100*np.mean((predicted_label == test_label).astype(float))) + '%'