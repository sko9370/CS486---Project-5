# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = list(trainingData[0].keys()) # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        "*** YOUR CODE HERE ***"
        ######################### TRAINING ###########################
        cweights = []
        for c in Cgrid:
            self.initializeWeightsToZero()
            for iteration in range(self.max_iterations):
                print("Starting iteration ", iteration, "...")
                for i in range(len(trainingData)):
                    instance = trainingData[i]
                    vectors = util.Counter()
                    for l in self.legalLabels:
                        vectors[l] = self.weights[l] * instance
                    guess = vectors.argMax()

                    if guess != trainingLabels[i]:
                        # calculate tau
                        tauTop = ((self.weights[guess]-self.weights[trainingLabels[i]]) * instance ) + 1.0
                        tauBottom = 2*(instance*instance)
                        tauMin = min(tauTop/tauBottom, c)
                        tauInv = 1/tauMin
                        instance.divideAll(tauInv)
                        # add to correct label
                        self.weights[trainingLabels[i]] += instance
                        # subtract from wrong label
                        self.weights[guess] -= instance
            cweights.append(self.weights)
        ########################## EVALUATION #############################
        maxAccuracy = 0
        maxIndex = 0
        for i in range(len(cweights)):
            correct = len(validationData)
            total = len(validationData)

            self.weights = cweights[i]
            guesses = self.classify(validationData)
            for j in range(len(guesses)):
                if guesses[j] != validationLabels[j]:
                    correct -= 1
            if correct/total > maxAccuracy:
                maxAccuracy = correct/total
                maxIndex = i
                print("!!! maxAccuracy: " + str(maxAccuracy))
                print("!!! maxIndex: " + str(maxIndex))
        self.weights = cweights[maxIndex]

        """
        for iteration in range(self.max_iterations):
            print("Starting iteration ", iteration, "...")

            CWeights = []

            for C in Cgrid:

                weights = {}
                for label in self.legalLabels:
                    weights[label] = util.Counter()

                for i in range(len(trainingData)):
                    instance = trainingData[i]
                    vectors = util.Counter()
                    for l in self.legalLabels:
                        vectors[l] = self.weights[l] * instance
                    guess = vectors.argMax()

                    # if guessed label is different from actual label
                    if guess != trainingLabels[i]:
                        # calculate tau
                        tau = (( (self.weights[guess]-self.weights[trainingLabels[i]]) * instance ) + 1.0)/(2*(instance*instance))
                        tauMin = min(tau, C)
                        tauInv = 1/tauMin
                        instance.divideAll(tauInv)
                        # add to correct label
                        self.weights[trainingLabels[i]] += instance
                        # subtract from wrong label
                        self.weights[guess] -= instance

                # store weights with C value
                CWeights.append(weights)
                print("appended new cweight")

            # evaluate accuracy for each C using validationData
            maxAccuracy = 0
            maxIndex = 0
            for i in range(len(CWeights)):
                correct = len(validationData)
                total = len(validationData)

                # set self.weights for testing using classify
                self.weights = CWeights[i]
                guesses = self.classify(validationData)
                for j in range(len(guesses)):
                    if guesses[j] != validationLabels[j]:
                        correct -= 1

                # better accuracy
                if correct/total > maxAccuracy:
                    maxAccuracy = correct/total
                    print("better accuracy " + str(maxAccuracy))
                    maxIndex = i
                    print("accuracy index " + str(maxIndex))

            # set those weights to self.weights
            self.weights = CWeights[maxIndex]
            """

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses
