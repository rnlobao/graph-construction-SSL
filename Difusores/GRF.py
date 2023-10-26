import numpy as np
from scipy.linalg import solve
import time

class GRF:
    def __init__(self, priors):
        self.priors = priors

    def classify(self, weightedMatrix, Y, l):
        startTime = time.time()
        D = self.computeDiagonalValues(weightedMatrix)
        Luu = self.computeLuu(weightedMatrix, D, l)
        Lul = self.computeLul(weightedMatrix, D, l)
        Yl = self.computeYl(Y, l)
        Fu = solve(Luu, -np.dot(Lul, Yl))
        totalTime = (time.time() - startTime)

        return totalTime, self.ClassMassNormalization(Fu, self.priors, 0)

    @staticmethod
    def computeDeltaUU(weightedMatrix, l):
        n = len(weightedMatrix)
        eps = 1.01
        DeltaUU = np.zeros((n - l, n - l))
        for i in range(l, n):
            sum = 0
            for j in range(l):
                sum += weightedMatrix[i][j]
            for j in range(l, n):
                DeltaUU[i - l, j - l] = -weightedMatrix[i][j]
                sum += weightedMatrix[i][j]
            DeltaUU[i - l, i - l] = eps * sum
        return DeltaUU

    @staticmethod
    def computeLuu(weightedMatrix, D, l):
        n = len(weightedMatrix)
        Luu = np.zeros((n - l, n - l))
        M = np.zeros(n - l)
        for i in range(l, n):
            M[i - l] = 1.0 / np.sqrt(D[i])
        for i in range(n - l):
            Luu[i, i] = 1.01
            for j in range(n - l):
                if i != j and weightedMatrix[i + l][j + l] != 0:
                    value = -weightedMatrix[i + l][j + l] * M[i] * M[j]
                    Luu[i, j] = value
        return Luu

    @staticmethod
    def computeLul(weightedMatrix, D, l):
        n = len(weightedMatrix)
        Lul = np.zeros((n - l, l))
        M = np.zeros(n)
        for i in range(n):
            M[i] = 1.0 / np.sqrt(D[i])
        for i in range(n - l):
            for j in range(l):
                if weightedMatrix[i + l][j] != 0:
                    value = -weightedMatrix[i + l][j] * M[i + l] * M[j]
                    Lul[i, j] = value
        return Lul

    @staticmethod
    def computeWUL(weightedMatrix, l):
        n = len(weightedMatrix)
        WUL = np.zeros((n - l, l))
        for i in range(l, n):
            for j in range(l):
                WUL[i - l, j] = weightedMatrix[i][j]
        return WUL

    @staticmethod
    def computeYl(Y, l):
        c = Y.shape[1]
        Yl = np.zeros((l, c))
        for i in range(l):
            for j in range(c):
                Yl[i, j] = Y[i, j]
        return Yl
