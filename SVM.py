import numpy as np 

import numpy 



path = r'//Users/kennedymaturure/Library/Mobile Documents/com~apple~CloudDocs/Documents/Machine learning /Machine learning project/dataset'
filename = "/diabetes_binary_health_indicators_BRFSS2015.csv"
inputData = numpy.loadtxt(path+filename, dtype = str, delimiter = ',')[1:] #remove title element 

numpy.random.seed(0)

# Separate based on the whether diabetic or non-diabetic
inputData = numpy.array(inputData, dtype = numpy.float64)
data_0 = inputData[inputData[:, 0] == 0] # Shape is (218334, 22)
data_1 = inputData[inputData[:, 0] == 1] # Shape is (35346, 22)

#Shuffle Observations
numpy.random.shuffle(data_0)

num_rows = data_1.shape[0]

dataSet = numpy.vstack((data_1, data_0[0:num_rows, :]))

numpy.random.shuffle(dataSet) 

# Number of Eigenvectors used
numEig = 10

#Pre-process Data
#Separate data and goal values
Y = numpy.atleast_2d(dataSet[:,0]).T 
X = dataSet[:, 1:]

Y = numpy.array(Y, dtype=numpy.float64)
X = numpy.array(X, dtype=numpy.float64)

# Separate into training and validation data
# Split array, 2/3 for training, 1/3 for validation
tX = X[:(len(X)*2//3)] 
vX = X[(len(X)*2//3):] 

# Goal Values Split
tY = Y[:(len(Y)*2//3)]
vY = Y[(len(Y)*2//3):]

 # ZScore/Standardize the data using Training Data
dmean = numpy.mean(tX, axis = 0, keepdims = True)
dstd = numpy.std(tX, axis = 0, keepdims = True)
tX = (tX-dmean)/dstd
vX = (vX-dmean)/dstd

# compute covariance matrix
cov = numpy.cov(tX, ddof = 1, rowvar = False) #rowvar = false for features(col), true for observations(rows)

# get the eigenvalues of the covariance matrix using singular value decomposition
[eigvals, eigvecs] = numpy.linalg.eigh(cov)

# PCA 10 most relevant eigenvectors, these are the last values in the eigenvectors array by eigh. A k-value of 10 contains 66% of eigenvalues
w = eigvecs[:,-20:] # Shape is (21, 20)

# project data onto PCA coordinate system
tX = numpy.dot(tX , w)  # Shape is (169120, 10)
vX= numpy.dot (vX , w)

# Parameters for SVM
C = 1.0
tol = 1e-4
max_passes = 5

# Initialize variables for the training process
m, n = tX.shape
alpha = np.zeros(m)
b = 0
passes = 0

# Precompute the linear kernel matrix for training data
kernel_matrix = np.dot(tX, tX.T)

# Sequential Minimal Optimization (SMO) algorithm with precomputed kernel matrix
while passes < max_passes:
    num_changed_alphas = 0
    for i in range(m):
        Ei = np.sum(alpha * tY * kernel_matrix[i, :]) + b - tY[i]
        if (tY[i] * Ei < -tol and alpha[i] < C) or (tY[i] * Ei > tol and alpha[i] > 0):
            # Randomly select a second sample j different from i
            j = np.random.choice([x for x in range(m) if x != i])
            
            # Ensure j is within the valid range
            if j < m:
                Ej = np.sum(alpha * tY * kernel_matrix[j, :]) + b - tY[j]
                # Rest of the code remains the same
                alpha_i_old , alpha_j_old = alpha[i] , alpha[j]
            else:
                continue  # Skip this iteration if j is out of bounds

            # Compute L and H
            if tY[i] == tY[j]:
                L = max(0, alpha[j] + alpha[i] - C)
                H = min(C, alpha[j] + alpha[i])
            else:
                L = max(0, alpha[j] - alpha[i])
                H = min(C, C + alpha[j] - alpha[i])
            
            if L == H:
                continue

            eta = 2 * kernel_matrix[i, j] - kernel_matrix[i, i] - kernel_matrix[j, j]
            if eta >= 0:
                continue

            # Update alpha[j]
            alpha[j] -= tY[j] * (Ei - Ej) / eta
            alpha[j] = np.clip(alpha[j], L, H)

            if abs(alpha[j] - alpha_j_old) < tol:
                continue

            # Update alpha[i]
            alpha[i] += tY[i] * tY[j] * (alpha_j_old - alpha[j])

            # Update bias term b
            b1 = b - Ei - tY[i] * (alpha[i] - alpha_i_old) * kernel_matrix[i, i] - tY[j] * (alpha[j] - alpha_j_old) * kernel_matrix[i, j]
            b2 = b - Ej - tY[i] * (alpha[i] - alpha_i_old) * kernel_matrix[i, j] - tY[j] * (alpha[j] - alpha_j_old) * kernel_matrix[j, j]
            b = b1 if 0 < alpha[i] < C else b2 if 0 < alpha[j] < C else (b1 + b2) / 2

            num_changed_alphas += 1

    passes = passes + 1 if num_changed_alphas == 0 else 0

# Compute the weights from the learned alphas
weights = np.sum((alpha * tY)[:, None] * tX, axis=0)

# Predict function
def predict(X, weights, b):
    return np.sign(np.dot(X, weights) + b)

# Predict on the training data
y_train_pred = predict(tX, weights, b)

# Predict on the validation data
y_pred = predict(vX, weights, b)

Trainaccuracy = numpy.mean(y_train_pred == tY) 
print(f'Training Accuracy: { Trainaccuracy}')
valaccuracy = numpy.mean(y_pred.ravel() == vY.ravel())
print(f'Validation Accuracy: { valaccuracy}')

tp = numpy.sum((vY.ravel() == 1) & (y_pred.ravel() == 1))  # True Positives
fp = numpy.sum((vY.ravel() == 0) & (y_pred.ravel() == 1))  # False Positives
fn = numpy.sum((vY.ravel() == 1) & (y_pred.ravel() == 0))  # False Negatives

# Avoid division by zero
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f" Precision: {precision}")
print(f" Recall: {recall}")
print(f" F1-Score: {f_measure}")



