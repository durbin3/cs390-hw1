
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
np.set_printoptions(suppress=True)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
# ALGORITHM = "custom_net"
ALGORITHM = "tf_net"





class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1, layers=2, activation='sigmoid'):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.layers = np.zeros(layers)

        self.weights = []
        W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)
        self.weights.append(W1)
        for i in range(layers-2):
            self.weights.append(np.random.randn(self.neuronsPerLayer, self.neuronsPerLayer))
        self.weights.append(W2)
        self.activation = activation

    # Activation function.
    def __activation(self, x):
        if self.activation == 'sigmoid': return 1/(1+np.exp(-x))
        return max(0,x)

    # Activation prime function.
    def __activationDerivative(self, x):
        if self.activation == 'sigmoid':
            sig = self.__activation(x)
            return sig*(1-sig)
        if x > 0: return 1
        return 0

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
        print("Training")
        X = xVals
        Y = yVals
        for i in range(epochs):
            print("Epoch: ", i)
            if minibatches:
                xBatch = self.__batchGenerator(xVals,mbs)
                yBatch = self.__batchGenerator(yVals,mbs) 
                for x in xBatch:
                    Y = next(yBatch)
                    self.__backprop(x,Y)
            else:
                self.__backprop(X,Y)

            
    # Forward pass.
    def __forward(self, input):
        layers = []
        xVals = input
        for i in range(len(self.layers)):
            weights = self.weights[i]
            comp_layer = self.__activation(np.dot(xVals, weights))
            xVals = comp_layer
            layers.append(comp_layer)
        # layer1 = self.__sigmoid(np.dot(input, self.W1))
        # layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layers

    def __backprop(self, X, Y):
        layers = self.__forward(X)
        deltas = []
        num_layers = len(layers)
        l = num_layers - 1
        lastLayer = layers[l]
        error = (lastLayer - Y)
        for i in range(num_layers):
            l = num_layers-i-1
            layer = layers[l]
            if l == 0: break
            delta = error * self.__activationDerivative(layers[l-1].dot(self.weights[l]))
            deltas.append(delta)
            error = delta.dot(self.weights[l].T)
        firstDelta = error * self.__activationDerivative(X.dot(self.weights[l]))
        deltas.append(firstDelta)

        adjustments = []
        l = len(deltas)
        xVals = X
        for i in range(l):
            l -= 1
            adj = xVals.T.dot(deltas[l])*self.lr
            self.weights[i] -= adj
            xVals = layers[i]


        # l1,l2 = self.__forward(X)
        # l2_delta = (l2-Y)*(l2*(1-l2))
        # l1_delta = (l2_delta.dot(self.W2.T))*(l1*(1-l1))
        # l1_adj = (X.T.dot(l1_delta))*self.lr
        # l2_adj = (l1.T.dot(l2_delta))*self.lr
        # self.W1 -= l1_adj
        # self.W2 -= l2_adj

    # Predict.
    def predict(self, xVals):
        layers = self.__forward(xVals)
        return layers[len(layers)-1]

    # Run
    def run(self, xTest):
        preds = []
        for entry in xTest:
            output = self.predict(entry)
            pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            pred[np.argmax(output)] = 1
            preds.append(pred)
            
        return np.array(preds)

# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    xTrain, xTest = xTrain / 255.0, xTest / 255.0
    xTrain = xTrain.reshape(xTrain.shape[0], 28*28)
    xTest = xTest.reshape(xTest.shape[0], 28*28)
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building Custom_NN.")
        neurons = 15
        nn = NeuralNetwork_2Layer(xTrain.shape[1],yTrain.shape[1],neurons,.0001, layers=2) 
        print("Training Custom NN")
        nn.train(xTrain,yTrain,epochs=250,minibatches=False)        
        return nn
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        model = tf.keras.models.Sequential([
              tf.keras.layers.Dense(512, activation='relu'),
              tf.keras.layers.Dense(256, activation='sigmoid'),
              tf.keras.layers.Dense(128, activation='sigmoid'),
              tf.keras.layers.Dense(64, activation='sigmoid'),
              tf.keras.layers.Dense(32, activation='relu'),
              tf.keras.layers.Dropout(.2),
              tf.keras.layers.Dense(10)
        ])
        loss = tf.keras.losses.MeanSquaredError()
        model.compile(optimizer='adam', loss=loss,metrics=['accuracy'])
        model.fit(xTrain, yTrain, epochs=20)
        return model
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        return model.run(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        return preds
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    con_mat = np.zeros((10,10))
    for i in range(preds.shape[0]):
        pred = np.argmax(preds[i])
        sol = np.argmax(yTest[i])
        if pred == sol:   
            acc = acc + 1
        con_mat[pred][sol] += 1

    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print("Confusion Matrix: \n", con_mat)



#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()