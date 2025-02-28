import numpy 
import os 
from collections import Counter 




path = r'/Users/kennedymaturure/Library/Mobile Documents/com~apple~CloudDocs/Documents/Machine learning /Machine learning project/dataset'
filename = "/diabetes_binary_health_indicators_BRFSS2015.csv"
inputData = numpy.loadtxt(path+filename, dtype = str, delimiter = ',')[1:] #remove title element 

#Pre-process Data
#Separate data and goal values
Y = numpy.atleast_2d(inputData[:,0]).T #Shape is (253680, 1)
X = numpy.atleast_2d(inputData[:, 1:]) # Shape is (253680, 21)

Y = numpy.array(Y, dtype=numpy.float64)
X = numpy.array(X, dtype=numpy.float64)

# Separate into training and validation data
# Split array, 2/3 for training, 1/3 for validation
x_train = X[:(len(X)*2//3)]  # Shape is (169120, 21)
x_val = X[(len(X)*2//3):]  # Shape is (84560, 21)

# Goal Values Split
y_train = Y[:(len(Y)*2//3)]  # Shape is (169120, 1)
y_val = Y[(len(Y)*2//3):]  # Shape is (169120, 1)


# ZScore/Standardize the data using Training Data
dmean = numpy.mean(x_train, axis = 0, keepdims = True)
dstd = numpy.std(x_train, axis = 0, keepdims = True)
x_train = (x_train-dmean)/dstd
x_val = (x_val-dmean)/dstd

# compute covariance matrix
cov = numpy.cov(x_train, ddof = 1, rowvar = False) #rowvar = false for features(col), true for observations(rows)

# get the eigenvalues of the covariance matrix using singular value decomposition
[eigvals, eigvecs] = numpy.linalg.eigh(cov)

# PCA 10 most relevant eigenvectors, these are the last values in the eigenvectors array by eigh. A k-value of 10 contains 66% of eigenvalues
w = eigvecs[:,-10:] # Shape is (21, 10)

# project data onto PCA coordinate system
 # Shape is (169120, 10)

x_train_pca = numpy.dot(x_train, w)
x_val_pca = numpy.dot(x_val, w)

# Logistic Regression implementation (using gradient descent)
def logistic_regression(X, y, learning_rate=0.1, epochs=1000):
    m, n = X.shape
    theta = numpy.zeros(n)
    y = y.flatten()
    for _ in range(epochs):
        predictions = 1 / (1 + numpy.exp(-X.dot(theta)))  # Sigmoid function
        gradient = X.T.dot(predictions - y) / m
        theta -= learning_rate * gradient  # Update weights
    return theta

def predict_logistic_regression(X, theta):
    return (1 / (1 + numpy.exp(-X.dot(theta))) >= 0.5).astype(int)

def naive_bayes_predict(X_train, y_train, X_test):
    classes = numpy.unique(y_train)
    priors = {}
    cond_prob = {}
    
    for cls in classes:
        cls_index = numpy.where(y_train == cls)[0]
        x_cls = X_train[cls_index]
        
        priors[cls] = len(cls_index) / len(y_train)
        cond_prob[cls] = numpy.mean(x_cls, axis=0)
    
    y_pred = []
    for x in X_test:
        posteriors = {}
        for cls in classes:
            prior_log = numpy.log(priors[cls])
            prob_cls = numpy.clip(cond_prob[cls], 1e-3, 1 - 1e-3)
            prob_log = numpy.sum(x * numpy.log(prob_cls) + (1 - x) * numpy.log(1 - prob_cls))
            posteriors[cls] = prior_log + prob_log
        y_pred.append(max(posteriors, key=posteriors.get))
    
    return numpy.array(y_pred)

def hard_voting(models, X):
    y_pred = []
    
    for x in X:
        # Get the predictions from all models for this sample
        votes = [model(x.item) for model in models]
        
        # Get the most common class (majority vote)
        prediction = numpy.bincount(votes).argmax()  # Find the class with the most votes
        y_pred.append(prediction)
    
    return numpy.array(y_pred) 


def soft_voting(models, X):
    y_pred = []
    
    for x in X:
        # Get the probabilities from all models for this sample
        probas = numpy.array([model(x) for model in models])
        
        # Calculate the average probability for each class
        avg_proba = numpy.mean(probas, axis=0)
        
        # Choose the class with the highest average probability
        prediction = numpy.argmax(avg_proba)
        y_pred.append(prediction)
    
    return numpy.array(y_pred)

log_reg_model = logistic_regression(x_train, y_train)
naive_bayes_model = naive_bayes_predict(x_train, y_train, x_val)

# Create ensemble model
models = [log_reg_model, naive_bayes_model]

# Hard voting
y_pred_hard = hard_voting(models, x_val)

# Soft voting (assuming models output probabilities)
y_pred_soft = soft_voting(models, x_val)

# Evaluate both hard voting and soft voting
def evaluate(y_true, y_pred):
    accuracy = numpy.mean(y_true == y_pred)
    precision = numpy.sum((y_true == 1) & (y_pred == 1)) / numpy.sum(y_pred == 1)
    recall = numpy.sum((y_true == 1) & (y_pred == 1)) / numpy.sum(y_true == 1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1_score

# Evaluate hard voting model
accuracy_hard, precision_hard, recall_hard, f1_hard = evaluate(y_val, y_pred_hard)

# Evaluate soft voting model
accuracy_soft, precision_soft, recall_soft, f1_soft = evaluate(y_val, y_pred_soft)

print("Hard Voting - Accuracy:", accuracy_hard)
print("Hard Voting - Precision:", precision_hard)
print("Hard Voting - Recall:", recall_hard)
print("Hard Voting - F1 Score:", f1_hard)

print("\nSoft Voting - Accuracy:", accuracy_soft)
print("Soft Voting - Precision:", precision_soft)
print("Soft Voting - Recall:", recall_soft)
print("Soft Voting - F1 Score:", f1_soft)