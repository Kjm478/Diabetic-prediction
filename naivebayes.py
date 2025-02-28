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


def NaiveBayes( x_train, x_val , y_train): 
    #compute priors and conditional probabilities 
    classes = numpy.unique(y_train)
    priors = {} 
    cond_prob = {} 
    ep = 1e-9


    total_samples = len(y_train)
    class_weights = {cls: total_samples / numpy.sum(y_train == cls) for cls in classes}
    for cls in classes: 
        cls_index = numpy.where(y_train == cls)[0]
        x_cls = x_train[cls_index]
        
        #Gausian probabilities
        priors[cls] = len(cls_index)   / total_samples * class_weights[cls]
        mean = numpy.mean(x_cls , axis=0 )
        var = numpy.var(x_cls, axis= 0) + ep # add a smoothing 
        cond_prob[cls] = (mean , var )

    # validation samples  
    
    y_pred = []
    for x in x_val: 
        posteriors = {} 
        for cls in classes: 
            prior_log = numpy.log(priors[cls])
            mean , var = cond_prob[cls]
            likelihood = -0.5 * numpy.sum(((x - mean) ** 2) / var + numpy.log(2 * numpy.pi * var))
            posteriors[cls] = prior_log + likelihood
        y_pred.append(max(posteriors, key=posteriors.get))
    return numpy.array(y_pred)


if __name__ == '__main__':
    #Prediction
    y_pred = NaiveBayes(tX, vX ,tY)
    #compute accuracy 
    Trainaccuracy = numpy.mean(y_pred == tY) 
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









