import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import sys

X = np.genfromtxt(sys.argv[1], delimiter=",")
K = 5  # number of clusters

# Initialization: 5 first element to the list
centerslist = X[:K, :]


# For each data return the class of hte closet centroid
# Return array of class
def update_class(data):
    return np.apply_along_axis(closest_centroid, 1, data)


# Return the class of the closest centroid
def closest_centroid(x):
    distances = [np.linalg.norm(x - center) for center in centerslist]
    return distances.index(min(distances))


# Update the centroid based on the class of data
def update_centerslist(data, y):
    for i in range(K):
        sub_X = data[y == i, :]
        centerslist[i] = sub_X.mean(0)


# K-means
def KMeans(data):
    # perform the algorithm with 5 clusters and 10 iterations...you may try others for testing purposes, but submit 5
    # and 10 respectively

    for i in range(10):
        # Update class of each data
        y = update_class(data)

        # update centroid
        update_centerslist(data, y)

        filename = "centroids-" + str(i + 1) + ".csv"  # "i" would be each iteration
        np.savetxt(filename, centerslist, delimiter=",")


# EM Gaussian mixture models
def EMGMM(data):
    n = data.shape[0]
    d = data.shape[1]

    # Initialization
    pi = [1 / K] * K
    mu = data[:K, :]            # (K, d) matrix
    sigma = [np.identity(d)]*K  # (K, (d, d)) list

    for it in range(10):
        # E-step
        phi = np.zeros((n, K))
        update_phi(data, mu, n, phi, pi, sigma)

        # M-step
        nk = np.sum(phi, axis=0)
        pi = np.true_divide(nk, n)
        mu = update_mu(data, n, nk, phi)
        update_sigmas(d, data, mu, n, nk, phi, sigma)

        print_value(it, mu, pi, sigma)


def update_sigmas(d, data, mu, n, nk, phi, sigma):
    for k in range(K):
        sigma[k] = np.zeros((d, d))
        for i in range(n):
            xi_mu = [data[i][j] - mu[k][j] for j in range(d)]
            sigma[k] += np.multiply(prod_row_col(xi_mu, xi_mu), phi[i][k])
        sigma[k] = sigma[k] / nk[k]


def update_mu(data, n, nk, phi):
    mu = [sum([phi[i][k] * data[i] for i in range(n)]) / nk[k] for k in range(K)]  # (K, d) matrix
    return mu


def update_phi(data, mu, n, phi, pi, sigma):
    for i in range(n):
        phi[i] = [pi[k] * sp.stats.multivariate_normal(mu[k], sigma[k]).pdf(data[i]) for k in range(K)]
        phi[i] = phi[i] / sum(phi[i])


# Create *.csv files for pi, mu and sigmas
def print_value(it, mu, pi, sigma):
    print_pi(it, pi)
    print_mu(it, mu)
    print_sigmas(it, sigma)


def print_pi(it, pi):
    filename = "pi-" + str(it + 1) + ".csv"
    np.savetxt(filename, pi, delimiter=",")


def print_mu(it, mu):
    filename = "mu-" + str(it + 1) + ".csv"
    np.savetxt(filename, mu, delimiter=",")  # this must be done at every iteration


def print_sigmas(it, sigma):
    for k in range(K):  # k is the number of clusters
        filename = "Sigma-" + str(k + 1) + "-" + str(it + 1) + ".csv"  # this must be done 5 times (or the number
        # of clusters) for each iteration
        np.savetxt(filename, sigma[k], delimiter=",")


# Return the product of a row of dimension d times a column of dimension n
# return a (n, d) matrix
def prod_row_col(row, col):
    d = len(row)
    n = len(col)
    result = [[row[j]*col[i] for j in range(d)] for i in range(n)]
    return result


KMeans(X)
EMGMM(X)