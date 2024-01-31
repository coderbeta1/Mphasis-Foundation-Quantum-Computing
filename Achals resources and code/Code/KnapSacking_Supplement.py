import numpy as np
from numpy import random

# weights = random.randint(1,100,size=(5))
values = random.randint(1,100,size=(5))
weights=[5,5,5,5,5]
print(weights)
population = ['10101','01011','00011','00000','11111']

def calcWeight(weights, solution):
    weightSum=0
    for weight in range(len(weights)):
        if(population[weight]=='1'):
            weightSum+=weights[weight]
    return weightSum

def repairFunc(weights, solution, knapsackSize):
    knapsack_full = False
    weightSum= calcWeight(weights, solution)

    if weightSum>knapsackSize:
        knapsack_full=True
    
    while(knapsack_full==True):
        randomWeightIndex = random.randint(1,len(weights)+1)-1
        check = solution[randomWeightIndex]

        if(check=='1'):
            solution = solution[:randomWeightIndex] + solution[randomWeightIndex:].replace(solution[randomWeightIndex], '0', 1)
            weightSum-=weights[randomWeightIndex]
        if(weightSum<knapsackSize):
            break
    
    while(knapsack_full==False):
        randomWeightIndex = random.randint(1,len(weights))-1
        check = solution[randomWeightIndex]
        if(check=='0'): 
            solution = solution[:randomWeightIndex] + solution[randomWeightIndex:].replace(solution[randomWeightIndex], '1', 1)
            weightSum+=weights[randomWeightIndex]
        if(weightSum>knapsackSize):
            break
    return

print(calcWeight(weights, solution=population[0]))
repairFunc(weights,solution=population[0], knapsackSize=25)
print(calcWeight(weights, solution=population[0]))