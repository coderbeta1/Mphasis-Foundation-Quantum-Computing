
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from random import random
 
# set width of bar 
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 

first = pd.read_csv("ClassicalQuantum_updated.csv")
print(first)

# set height of bar
IT = list(first['Anneling Time'])[:9]
ECE = list(first['Anneling Time'])[9:]

print(len(IT), len(ECE))

# Set position of bar on X axis 
br1 = np.arange(len(IT)) 
br2 = [x + barWidth for x in br1] 

# Make the plot
plt.bar(br1, IT, color ='green', width = barWidth, 
        edgecolor ='grey', label ='Quantum') 
plt.bar(br2, ECE, color ='blue', width = barWidth, 
        edgecolor ='grey', label ='Classical')  
 
# Adding Xticks 
print("Here")
plt.xlabel('Simulations', fontweight ='bold', fontsize = 15) 
plt.ylabel('Runtime (QPU for Quantum)', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(IT))], 
        list(first['Disrupted File Inventory ID'])[:9], rotation = 90)
 
plt.legend()
plt.show()