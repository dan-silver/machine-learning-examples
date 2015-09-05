import numpy as np
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

# data from http://archive.ics.uci.edu/ml/datasets/Heart+Disease
data =  np.loadtxt(fname = "heart_disease.csv", delimiter = ',')

# Extract the last column, which is the result we're trying to predict
X, Y = data[:, :-1], data[:, -1]

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=50)

# Scale the data from -1 to +1
x_train = scale(x_train)
x_test = scale(x_test)

# Specify the classifier will be a SVM. Note by changing this one line, we can swap in and out different classifiers
clf = svm.SVC()

# Train the classifier - this is what takes the most time
clf.fit(x_train, y_train) 

# Given our trained classifier, make some predictions
y_predict = clf.predict(x_test)

print accuracy_score(y_test, y_predict)