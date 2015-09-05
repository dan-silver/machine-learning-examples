import numpy as np
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

# data from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
data =  np.loadtxt(fname = "winequality-white.csv", delimiter = ',')

# Extract the last column, which is the result we're trying to predict
X, Y = data[:, :-1], data[:, -1]

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=50)

# Scale the data from -1 to +1
x_train = scale(x_train)
x_test = scale(x_test)


# Different C and gamma values that will be tested
C_2d_range = [1e-2, 1, 1e2, 1e3]
gamma_2d_range = [1e-1, 1, 1e1]


# Try all the combinations, train, test, and print the accuracy
for C in C_2d_range:
  for gamma in gamma_2d_range:
    clf = svm.SVC(C=C, gamma=gamma)
    clf.fit(x_train, y_train)

    # Given our trained classifier, make some predictions
    y_predict = clf.predict(x_test)

    print (C, gamma), accuracy_score(y_test, y_predict)