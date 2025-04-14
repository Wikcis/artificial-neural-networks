import random
import matplotlib.pyplot as plt
import numpy as np

inputSize = 2


def pickRandomTrainingVector(numberOfData, minValue, maxValue):
    return [(random.randint(minValue, maxValue), random.randint(minValue, maxValue)) for _ in range(numberOfData)]


def pickRandomReferenceVector(vector):
    tmpList = []
    for _ in vector:
        if _[0] < _[1]:
            tmpList.append(1)
        else:
            tmpList.append(-1)
    return tmpList


def initializeWeights():
    return [random.uniform(-1, 1) for _ in range(inputSize + 1)]


def activate(value):
    return 1 if value >= 0 else -1


def predict(weights, point):
    weightedSum = sum(point[j] * weights[j + 1] for j in range(len(point))) + weights[0]
    return weightedSum


def train(weights, point, referenceValue):
    activatedPredictedPointValue = activate(predict(weights, point))
    if activatedPredictedPointValue != referenceValue:
        error = referenceValue - activatedPredictedPointValue
        weights[0] += error

        for i in range(inputSize):
            weights[i + 1] += error * point[i]


def drawPlot(weights, x, y, title):
    x = np.array(x)
    y = np.array(y)

    if weights[2] != 0:
        a = -weights[1] / weights[2]
        b = -weights[0] / weights[2]
        line_x = np.linspace(min(x), max(x), 100)
        line_y = a * line_x + b
        plt.plot(line_x, line_y, '-r', label=title)

        above_line = y > a * x + b
        below_line = ~above_line

        plt.scatter(x[above_line], y[above_line], marker='o', color='red', label='Above')
        plt.scatter(x[below_line], y[below_line], marker='o', color='blue', label='Below')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid()
    plt.show()


def trainVector(weights, vector, referenceVector):
    for _ in range(1000):
        for o in range(len(vector)):
            train(weights, vector[o], referenceVector[o])


def main():
    trainingVector = [[-8, 20], [-6, -25], [-9, -10], [-6, -5], [-2, -10],
                      [1, -10], [2, 20], [5, 9], [4, -16], [4, 25]]
    referenceVector = [1, -1, 1, 1, -1, -1, 1, -1, -1, 1]
    weights = initializeWeights()

    trainVector(weights, trainingVector, referenceVector)

    x = [point[0] for point in trainingVector]
    y = [point[1] for point in trainingVector]
    drawPlot(weights, x, y, 'Prosta separująca grupy wybranych punktów')

    sizeOfRandomData = 20
    minValueForRandomData = -30
    maxValueForRandomData = 30

    randomTrainingVector = pickRandomTrainingVector(sizeOfRandomData, minValueForRandomData, maxValueForRandomData)
    randomReferenceVector = pickRandomReferenceVector(randomTrainingVector)
    weights = initializeWeights()
    trainVector(weights, randomTrainingVector, randomReferenceVector)

    x = [point[0] for point in randomTrainingVector]
    y = [point[1] for point in randomTrainingVector]
    drawPlot(weights, x, y, 'Prosta separująca grupy losowych punktów')

    test_data = [([-7, 15], 1), ([3, -5], -1), ([-2, 15], 1), ([2, 20], 1), ([0, -30], -1)]

    for points, target in test_data:
        prediction = activate(predict(weights, points))
        print(f"Input: {points}, Target: {target}, Prediction: {prediction}")


if __name__ == "__main__":
    main()