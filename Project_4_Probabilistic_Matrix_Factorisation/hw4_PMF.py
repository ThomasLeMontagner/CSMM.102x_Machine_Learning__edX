from __future__ import division
import numpy as np
import sys

train_data = np.genfromtxt(sys.argv[1], delimiter=",")

lam = 2
sigma2 = 0.1
d = 5
iterations = 50


# Convert the training data into a missing matrix
def get_missing_matrix(train_data):
    users = set(train_data[:, 0])
    objects = set(train_data[:, 1])

    # Create dictionary for User: [user, row number]
    users_dict = dict()
    i = 0
    for u in users:
        users_dict[u] = i
        i += 1

    # Create dictionary for Objects: [object, column number]
    objects_dict = dict()
    i = 0
    for o in objects:
        objects_dict[o] = i
        i += 1

    # Initialize matrix M with measured pairs
    M = np.zeros((len(users), len(objects)))
    for row in train_data:
        i = users_dict[row[0]]
        j = objects_dict[row[1]]
        r = row[2]  # rating
        M[i, j] = r

    return M


M = get_missing_matrix(train_data)
Nu = M.shape[0]
Nv = M.shape[1]

# for each ui, return list of column for which there is a rate (i.e. != 0)
def get_sigma_U(M):
    sigma_U = []
    for i in range(Nu):
        sigma_ui = []
        for j in range(Nv):
            if M[i][j] != 0:
                sigma_ui.append(j)
        sigma_U.append(sigma_ui)
    return sigma_U


# for each vi, return list of row for which there is a rate (i.e. != 0)
def get_sigma_V(M):
    sigma_V = []
    for j in range(Nv):
        sigma_vi = []
        for i in range(Nu):
            if M[i][j] != 0:
                sigma_vi.append(i)
        sigma_V.append(sigma_vi)
    return sigma_V


sigma_U = get_sigma_U(M)
sigma_V = get_sigma_V(M)


# Initialize the V matrix with Mu = 0, and sigma = I/lambda
def initialize(Nv):
    v0 = []
    for i in range(Nv):
        voi = np.random.normal(0, 1 / lam, d) # check that
        # voi =[1/lam]*d
        v0.append(voi)
    return v0


# Update U from V
def update_u(v):
    u = []
    for i in range(Nu):
        sum_v = np.zeros((d, d))
        for j in sigma_U[i]:
            sum_v += np.outer(np.transpose(v[j]), v[j])

        sum_Mv = [0]*d
        for j in sigma_U[i]:
            sum_Mv += np.dot(M[i][j], v[j])
        I = np.identity(d)
        ui = np.dot(np.linalg.inv(lam * sigma2 * I + sum_v), sum_Mv)
        u.append(ui)
    return u


# Update U from V
def update_u_old(v):
    u = []
    for i in range(Nu):
        sum_v = 0
        for j in sigma_U[i]:
            sum_v += v[j].dot(np.transpose(v[j]))

        sum_Mv = [0]*d
        for j in sigma_U[i]:
            sum_Mv += M[i][j] * v[j]
        ui = 1 / (lam * sigma2 + sum_v) * sum_Mv
        u.append(ui)
    return u

# Update V from U
def update_v(u):
    v = []
    for j in range(Nv):
        sum_u = np.zeros((d, d))
        for i in sigma_V[j]:
            sum_u += np.outer(np.transpose(u[i]), u[i])

        sum_Mu = [0]*d
        for i in sigma_V[j]:
            sum_Mu += np.dot(M[i][j], u[i])
        I = np.identity(d)
        vi = np.dot(np.linalg.inv(lam * sigma2 * I + sum_u), sum_Mu)
        v.append(vi)
    return v


def update_v_old(u):
    v = []
    for j in range(Nv):
        sum_Mu = [0]*d
        sum_u = 0
        for i in sigma_V[j]:
            sum_u += u[i].dot(np.transpose(u[i]))

        for i in sigma_V[j]:
            sum_Mu += M[i][j] * u[i]
        vi = 1 / (lam * sigma2 + sum_u) * sum_Mu
        v.append(vi)
    return v


# Calculate hte objective function
def obj_funct(U, V):
    L = 0
    for i in range(Nu):
        for j in sigma_U[i]:
            L -= (1 / (2 * sigma2)) * (M[i][j] - np.dot(U[i], np.transpose(V[j]))) ** 2

    L -= sum(lam / 2 * (np.linalg.norm(U[i]) ** 2) for i in range(Nu))
    L -= sum(lam / 2 * (np.linalg.norm(V[j]) ** 2) for j in range(Nv))
    return L


# Perform PMF algorithm
def PMF():
    L = []
    U_matrices = []
    V_matrices = []

    for i in range(iterations):
        # update U
        if i == 0:
            v0 = initialize(Nv)
            U = update_u(v0)
        else:
            U = update_u(V_matrices[i - 1])
        U_matrices.append(U)

        # Update V
        V = update_v(U_matrices[i])
        V_matrices.append(V)

        # Update objective function
        L.append(obj_funct(U, V))
    return L, U_matrices, V_matrices


# Assuming the PMF function returns Loss L, U_matrices and V_matrices (refer to lecture)
L, U_matrices, V_matrices = PMF()

np.savetxt("objective.csv", L, delimiter=",")

np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
np.savetxt("V-50.csv", V_matrices[49], delimiter=",")
