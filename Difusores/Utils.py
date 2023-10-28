import numpy as np

def computeGramMatrix(weightedMatrix, distanceMatrix, sigma):
    n = len(weightedMatrix)
    K = np.zeros((n, n))
    den = 2 * sigma * sigma
    for i in range(n):
        for j in range(i + 1, n):
            dist = distanceMatrix[i][j]
            weight = np.exp(-(dist ** 2) / den)
            K[i][j] = weight
            K[j][i] = weight
        K[i][i] = 1
    return K


def getClassPriors(inputLabels, l, c):
    examplesPerClass = examplesPerClass(inputLabels, l, c)
    priors = np.zeros(c, dtype=float)
    for ix in range(c):
        priors[ix] = (examplesPerClass[ix] * 1.0) / l
    return priors


def computeDiagonalValues(weightedMatrix):
    n = len(weightedMatrix)
    D = np.zeros(n)
    for i in range(n):
        for j in range(n):
            D[i] += weightedMatrix[i][j]
    return D

def normalizedLaplacian(weightedMatrix, D):
    n = len(weightedMatrix)
    L = np.zeros((n, n))
    M = np.zeros(n)
    for i in range(n):
        M[i] = D[i] ** -0.5
    for i in range(n):
        L[i, i] = 1.01
        for j in range(n):
            if i != j and weightedMatrix[i][j] != 0:
                L[i][j] = -weightedMatrix[i][j] * M[i] * M[j]
    return L

def orderedNumbers(n):
    orderedNumbers = list(range(n))
    return orderedNumbers
