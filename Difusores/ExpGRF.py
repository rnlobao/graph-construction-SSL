import numpy as np
import time
from . import GRF, ActualState, Utils

class ExpGRF:

    @staticmethod
    def propagateLabels(dataset, graph, pl, plID):
            tmpResponse = ExpGRF.computeGFHF(graph, 2, len(dataset), len(pl), plID)


    @staticmethod
    def computeGFHF(graph, nroDePossiveisLabels, lenDoDataset, numeroDeInstanciasRotuladas, split):
        startTime = time.time()
        D = Utils.computeDiagonalValues(graph.to_num)
        L = Utils.normalizedLaplacian(graph.to_numpy_matrix(), D)
        normLaplacian = L.tolist()
        orderedNumbers = Utils.orderedNumbers(lenDoDataset)
        print(orderedNumbers)
        labels = [0] * lenDoDataset

        ix = 0
        for i in originalOrder:
            labels[ix] = originalLabels[ix]
            orderedNumbers[ix] = i
            ix += 1

        gramMatrix = Utils.computeGramMatrix(graph.getWeightedMatrix(), graph.getDistanceMatrix(), graph.getSigma())

        timeLaplacian = (time.time() - startTime) / 1000.00

        split.sort()
        actualState = ActualState(graph.getData(), graph.getDistanceMatrix(), graph.getWeightedMatrix(), graph.getAdjacencyMatrix(), orderedNumbers, split, labels, gramMatrix, normLaplacian)
        actualState.reorder()

        inputLabels = actualState.getActualLabels()[:]
        for ix in range(numeroDeInstanciasRotuladas, lenDoDataset):
            inputLabels[ix] = -1

        classPriors = Utils.getClassPriors(inputLabels, numeroDeInstanciasRotuladas, nroDePossiveisLabels)
        Y = Utils.computeLabelMatrixGFHF(inputLabels, numeroDeInstanciasRotuladas, nroDePossiveisLabels, lenDoDataset)

        grf = GRF(classPriors)
        pairResult = grf.classify(actualState.getActualWeightedMatrix(), Y, numeroDeInstanciasRotuladas)
        output, timeGRF = pairResult[1], pairResult[0]

        finalResult = {}
        for i in range(numeroDeInstanciasRotuladas, lenDoDataset):
            idObj = actualState.getActualNumbers()[i]
            finalResult[idObj] = output[i - numeroDeInstanciasRotuladas]

        return timeLaplacian + timeGRF, finalResult
