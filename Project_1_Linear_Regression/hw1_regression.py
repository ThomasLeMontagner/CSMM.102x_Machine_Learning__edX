import numpy as np
import sys

lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter = ",")

## Solution for Part 1
def part1():
    ## Input : Arguments to the function
    ## Return : wRR, Final list of values to write in the file
    # from the lecture "Ridge Regression" in Week 2: wRR = (lambda*I + X'*X)-1 * X' * y
    I = np.identity(X_train.shape[1])
    # X_train_t = X_train.transpose()
    return np.linalg.inv(lambda_input*I + np.dot(X_train.T, X_train)).dot(X_train.T).dot(y_train)

wRR = part1()  # Assuming wRR is returned from the function
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file


## Solution for Part 2
def part2():
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file

    # From elcture 5 Bayesian Linear Regression:
    # Prior: (lambda*I + sigma-2*X'*X)-1 = E
    # Posterior: (lambda*I + sigma-2*(x0*x0' + X'*X))-1 = (E-1 + sigma-2*x0*x0')-1

    index_values = [] # output with index of the chosen values from X
    X = []
    for i in range(10):
        sigma02 = -1;  # dummy value, will necessarily be greater after
        chosen_rows = []
        print(i)
        for row in range(X_train.shape[0]):
            if not row in chosen_rows:
                I = np.identity(X_train.shape[1])
                X0 = Xi = X_train[row]
                if i != 0:
                    Xi = np.vstack((X, X0))
                E = np.linalg.inv(lambda_input*I + (1/sigma2_input)*(np.dot(Xi.T, Xi)))
                sigmai2 = sigma2_input + X0.dot(E.dot(X0.T))
                if sigmai2 > sigma02:
                    chosen_rows.append(row)
                    X = Xi

    return chosen_rows

active = part2()  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file