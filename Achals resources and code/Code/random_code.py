from numpy import random
import numpy as np
import math

# Generate Population
def generatePopulation(population_size, probabilities):
    population = []
    for i in range(population_size):
        currentString= ""
        for i in range(stringLength):
            prob=math.sin(math.radians(probabilities[i]))
            currentString+=str(np.random.choice([0,1], size=1, p=[prob, 1-prob])[0])
        population.append(currentString)
    return population

# Calculate Fitness
def calcFitness(string, weights, values, knapsackSize):
    weightFitness= 0
    valueFitness= 0

    for selection in range(len(string)):
        if string[selection]=='1':
            weightFitness+=weights[selection]
            valueFitness+=values[selection]
    
    if(weightFitness>knapsackSize):
        valueFitness=0
    return valueFitness

# Pick Best Solutions after Scoring
def sortAndBestSolution(population, weights, values, knapsackSize):
    scores={}
    for i in range(len(population)):
        scores[population[i]] = calcFitness(population[i], weights, values, knapsackSize)

    # Sort according to fitness
    scores=dict(sorted(scores.items(), key=lambda x:x[1], reverse=True))
    # print(scores)

    # Pick Best Solution
    bestString=list(scores.keys())[0]
    # print(bestString)
    return bestString, scores[bestString], scores

# Rotate probabilities towards the best string
def rotate(number, factor):
    if (number+factor < 0 or number+factor>90): return number
    return number + factor

def rotateProbability(string, probabilities, weights, values, maxWeight, scores):
    changeAmounts = []
    scoreKeys = list(scores.keys())
    for i in range(len(string)):
        changeValue = 0
        for j in range(4):
            if scoreKeys[j][i] == string[i]:
                changeValue += scores[scoreKeys[j]]
            else:
                changeValue -= scores[scoreKeys[j]]
        changeAmounts.append(changeValue)
    
    maxi= max(changeAmounts)
    mini= min(changeAmounts)
    for i in range(len(changeAmounts)):
        if maxi!= mini: changeAmounts[i] = (changeAmounts[i] - mini)/(maxi-mini) * (5-1) + 1
        else: changeAmounts[i] = 3
    

    sumWeights=0
    for i in range(len(weights)):
        if string[i]== '1': sumWeights+=weights[i]

    for i in range(len(probabilities)):
        if string[i]=='0':
            # if(sumWeights+weights[i]<maxWeight): probabilities[i] = rotate(probabilities[i], -2) 
            probabilities[i] = rotate(probabilities[i], -1 * changeAmounts[i])
        else:
            probabilities[i] = rotate(probabilities[i], changeAmounts[i])
    return probabilities

# Main Function
# weights = random.randint(1,10, size=300)
# values = random.randint(1,10, size=300)
weights = [95, 4, 60, 32, 23, 72, 80, 62, 65, 46]
values = [55, 10, 47, 5, 4, 50, 8, 61, 85, 87]
knapsackSize = 269

print("Weights: ", weights)
print("Values: ", values)
print("Knapsack Size: ", knapsackSize)
print()

stringLength = len(weights)
probabilities = [45 for i in range(stringLength)]

number_of_iterations= 100
population_size= 64
overallBestString = ""
overallBestValue = 0
for i in range(number_of_iterations):
    population = generatePopulation(population_size, probabilities)
    bestString, bestValue, scores = sortAndBestSolution(population, weights, values, knapsackSize)
    if(bestValue>overallBestValue):
        overallBestValue=bestValue
        overallBestString=bestString
    probabilities = rotateProbability(bestString, probabilities, weights, values, knapsackSize, scores)
    # print(probabilities)

print(probabilities)
print("Best String: ", overallBestString)
print("Best Value: ", overallBestValue)