import numpy as np
import time
import GRF, ActualState, Utils

class ExpGRF:
    positiveLabel = 1
    negativeLabel = 0

    @staticmethod
    def propagateLabels(dataset, graph, pl):
        classIds = dataset.getClassIds()
        totalTime = 0

        objects = dataset.getObjects()
        originalLabels = np.zeros(len(objects), dtype=int)
        originalOrder = np.zeros(len(objects), dtype=int)

        numberOfClass = len(classIds)
        numberOfPrelabeled = len(pl)
        numberOfObjects = len(objects)

        if numberOfClass == 2:  # Binary classification
            tmpResponse = (0.0, {})
            classIdOne = classIds.pop(0)
            classIdTwo = classIds.pop(0)

            prelabeledIds = np.zeros(len(pl), dtype=int)
            ix = 0

            for inst in objects:
                if inst.getTrueLabel() == classIdOne:
                    originalLabels[inst.getIndex()] = ExpGRF.positiveLabel
                    originalOrder[inst.getIndex()] = inst.getIndex()
                else:
                    originalLabels[inst.getIndex()] = ExpGRF.negativeLabel
                    originalOrder[inst.getIndex()] = inst.getIndex()

                if inst.isPreLabeled():
                    prelabeledIds[ix] = inst.getIndex()
                    ix += 1

            tmpResponse = ExpGRF.computeGFHF(graph, numberOfClass, numberOfObjects, numberOfPrelabeled, originalOrder, originalLabels, prelabeledIds)
            totalTime = tmpResponse[0]

            for key, value in tmpResponse[1].items():
                if value == ExpGRF.positiveLabel:
                    objects[key].setLabel(classIdOne)
                else:
                    objects[key].setLabel(classIdTwo)
            return totalTime

        for i in classIds:
            tmpResponse = (0.0, {})
            prelabeledIds = np.zeros(len(pl), dtype=int)
            ix = 0

            for inst in objects:
                if i == inst.getTrueLabel():
                    originalLabels[inst.getIndex()] = ExpGRF.positiveLabel
                    originalOrder[inst.getIndex()] = inst.getIndex()
                else:
                    originalLabels[inst.getIndex()] = ExpGRF.negativeLabel
                    originalOrder[inst.getIndex()] = inst.getIndex()

                if inst.isPreLabeled():
                    prelabeledIds[ix] = inst.getIndex()
                    ix += 1

            tmpResponse = ExpGRF.computeGFHF(graph, numberOfClass, numberOfObjects, numberOfPrelabeled, originalOrder, originalLabels, prelabeledIds)
            totalTime += tmpResponse[0]

            for key, value in tmpResponse[1].items():
                if value == ExpGRF.positiveLabel:
                    objects[key].setLabel(i)
        return totalTime

    @staticmethod
    def computeGFHF(graph, c, n, l, originalOrder, originalLabels, split):
        startTime = time.time()
        D = Utils.computeDiagonalValues(graph.getWeightedMatrix())
        L = Utils.normalizedLaplacian(graph.getWeightedMatrix(), D)
        normLaplacian = L.tolist()
        orderedNumbers = Utils.orderedNumbers(n)
        labels = [0] * n

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
        for ix in range(l, n):
            inputLabels[ix] = -1

        classPriors = Utils.getClassPriors(inputLabels, l, c)
        Y = Utils.computeLabelMatrixGFHF(inputLabels, l, c, n)

        grf = GRF(classPriors)
        pairResult = grf.classify(actualState.getActualWeightedMatrix(), Y, l)
        output, timeGRF = pairResult[1], pairResult[0]

        finalResult = {}
        for i in range(l, n):
            idObj = actualState.getActualNumbers()[i]
            finalResult[idObj] = output[i - l]

        return timeLaplacian + timeGRF, finalResult
