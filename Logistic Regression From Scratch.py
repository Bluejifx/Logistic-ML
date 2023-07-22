import numpy as np
import csv

class Logistic_Regression:

    # Initiating the hyperparameters of the model
    def __init__(self,learningRate,numOfIterations):
        self.learningRate = learningRate
        self.numOfIterations = numOfIterations



    # Fitting the dataset into the model class
    def fit(self,X,Y):

        # Getting the dimensions of the array
        self.m , self.n = X.shape

        # Initiating the weights and bias to 0
        self.w = np.array([np.zeros(self.m)])
        self.b = 0

        # Saving the dataset as attributes that are accessible in the object
        self.X = X
        self.Y = Y

        # Changing the number of iterations within each cycle
        for i in range(self.numOfIterations):
            self.update_weights()



    def update_weights(self):

        # Finding the value for the linear function and then using it as input for the sigmoid function
        LinearValue = np.dot(self.X.T,self.w.T) + self.b
        y_pred = 1/(1+np.exp(-LinearValue))

        # Applying gradient descent: finding the partial derivatives with respect to the weights and biases
        dw = np.dot((y_pred.T-self.Y),self.X.T)/self.m
        db = np.sum((y_pred.T-self.Y)) / self.m

        # Updating the weights and bias
        self.w = self.w - self.learningRate*dw
        self.b = self.b - self.learningRate*db



    # Predicting the final results after training the model
    def predict(self):
        LinearValue = np.dot(self.X.T,self.w.T) + self.b
        print(-LinearValue)
        y_pred = 1/(1+np.exp(-LinearValue))
        y_pred = np.where(y_pred > 0.55, 1, 0) # If the value of the predicted value is more than 0.5 then y_pred is 1, else, it will be 0

        return y_pred.T



# Determining the accuracy
def accuracy(predicted_values,actual_values):
    correct = 0
    for i in range(len(actual_values)):
        if predicted_values[i] == actual_values[i]:
            correct += 1

    return (correct/len(actual_values))




###############################################################################################




# Extracting the data from the CSV file into 2 numpy arrays
FieldNames = []
inputList = []
with open("diabetes.csv",'r') as file:
    reader = csv.reader(file)

    # Using a counter to not extract the field names
    n= 0
    for row in reader:
        if n > 0:
            inputList.append(row)
        if n == 0:
            FieldNames.append(row)
            n += 1


# Creating and adjusting the arrays
FieldNames = np.array(FieldNames).squeeze()
n = len(inputList)
inputList = np.array(inputList)
ArrayX = np.array(inputList[:n,:8],dtype="float64").T
ArrayY = np.array(inputList[:n,8:9],dtype="float64").T


# Assinging the hyperparameter values and fitting the dataset
Model = Logistic_Regression(0.001,1000)
Model.fit(ArrayX,ArrayY)


# Creating the prediction array and changing the array with the target feature values to a list
predictionArray = []
predictionArray.append(Model.predict())
predictionArray = np.array(predictionArray).squeeze(1).T.tolist()
print(predictionArray)
ArrayYTest = ArrayY.T.tolist()


# Computing the accuracy of the model and finding the feature that has the greatest effect on diabetes
print("Accuracy:",(accuracy(predictionArray,ArrayYTest)))
weightList = Model.w.tolist()
indexOfGreatestImpact = weightList.index(max(weightList))
print("The feature that has the greatest impact suggested by the model is:",FieldNames[indexOfGreatestImpact])
