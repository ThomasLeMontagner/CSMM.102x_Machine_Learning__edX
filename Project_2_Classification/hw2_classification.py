from __future__ import division
import numpy as np
import sys
import math

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])
X_test = np.genfromtxt(sys.argv[3], delimiter=",")

K = range(10)

## can make more functions if required


def pluginClassifier(X_train, y_train, X_test):
    # this function returns the required output
    # According to Lecture 7:
    # Use the data available to approximate P(Y=y) and P(X=x | Y=y)
    result = np.apply_along_axis(pluginClassifier_simple, axis=1, arr=X_test )
    return result


# Return the plugin Classifier vector for a single vector x
def pluginClassifier_simple(x):
    result = []
    priors = all_class_priors(y_train)
    cond_densities = all_class_cond_density(X_train, y_train)

    for k in K:
        mu_k = cond_densities[k][0]
        cov_k = cond_densities[k][1]
        x_mu_k = np.array([x - mu_k])
        proba = priors[k] * pow(np.linalg.det(cov_k), -0.5) * math.exp(-0.5 * x_mu_k.dot(np.linalg.inv(cov_k)).dot(x_mu_k.T))
        result.append(proba)

    norm_result = [p/sum(result) for p in result]

    return norm_result


def all_class_priors(y):
    dicts = {}
    #print("___Class priors___")
    for k in K:
        dicts[k] = class_priors(y, k)
        #print(k, ': ', dicts[k])
    return dicts


def all_class_cond_density(X, y):
    dicts = {}
    #print("___Class cond density___")
    for k in K:
        dicts[k] = class_cond_density(X, y, k)
    return dicts


# Return the Maximum Likelihood Estimate of y in the datasets
def class_priors(y, k):
    n = y.shape[0]
    return y.tolist().count(k) / n


def class_cond_density(X, y, k):
    X_train_y = X[y==k, :]
    mu_y = X_train_y.mean(0)
    cov_y = np.zeros((X.shape[1], X.shape[1]))
    #print(cov_y)
    for i in range(X_train_y.shape[0]):
        xi = X_train_y[i, :]
        m = [i*(xi - mu_y) for i in (xi - mu_y)]
        # print(m)
        cov_y = cov_y + m

    #print(cov_y)
    cov_y = cov_y / X_train_y.shape[0]
    return mu_y, cov_y


final_outputs = pluginClassifier(X_train, y_train, X_test)  # assuming final_outputs is returned from function

np.savetxt("probs_test.csv", final_outputs, delimiter=",")  # write output to file
