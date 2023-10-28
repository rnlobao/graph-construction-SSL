import numpy as np

class ActualState:
    def __init__(self, data, distanceMatrix, weightedMatrix, adjacencyMatrix, numbers, split, labels, gramMatrix=None, normalizedLaplacian=None):
        self.actualData = np.copy(data)
        self.actualWeightedMatrix = np.copy(weightedMatrix)
        self.actualAdjacencyMatrix = np.copy(adjacencyMatrix)
        self.actualNumbers = np.copy(numbers)
        self.actualSplit = np.copy(split)
        self.actualLabels = np.copy(labels)
        self.actualDistanceMatrix = np.copy(distanceMatrix)
        self.actualGramMatrix = np.copy(gramMatrix) if gramMatrix is not None else None
        self.actualNormalizedLaplacian = np.copy(normalizedLaplacian) if normalizedLaplacian is not None else None

    def getActualData(self):
        return self.actualData

    def getActualWeightedMatrix(self):
        return self.actualWeightedMatrix

    def getActualAdjacencyMatrix(self):
        return self.actualAdjacencyMatrix

    def getActualNumbers(self):
        return self.actualNumbers

    def getActualSplit(self):
        return self.actualSplit

    def getActualLabels(self):
        return self.actualLabels

    def getActualDistanceMatrix(self):
        return self.actualDistanceMatrix

    def getActualGramMatrix(self):
        return self.actualGramMatrix

    def getActualNormalizedLaplacian(self):
        return self.actualNormalizedLaplacian

    def reorder(self):
        l = len(self.actualSplit)
        self.actualSplit.sort()
        index = 0
        for j in range(l):
            if self.actualNumbers[index] != self.actualSplit[j]:
                self._changeLine(self.actualData, index, self.actualSplit[j])
                self._change(self.actualNumbers, index, self.actualSplit[j])
                self._change(self.actualWeightedMatrix, index, self.actualSplit[j])
                self._change(self.actualLabels, index, self.actualSplit[j])
                self._change(self.actualAdjacencyMatrix, index, self.actualSplit[j])
                
                if self.actualGramMatrix is not None:
                    self._change(self.actualGramMatrix, index, self.actualSplit[j])
                    self._change(self.actualNormalizedLaplacian, index, self.actualSplit[j])
            index += 1

    def _changeLine(self, matrix, i, j):
        matrix[i], matrix[j] = np.copy(matrix[j]), np.copy(matrix[i])

    def _change(self, array, i, j):
        array[i], array[j] = array[j], array[i]
