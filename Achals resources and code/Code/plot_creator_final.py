
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from random import random

data = pd.read_csv("Metrics.csv")
quantum = data.iloc[:9,:]
classical = data.iloc[9:,:]

# set width of bar 
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 
 
# set height of bar
IT = quantum['Exception Solution']
ECE = classical['Exception Solution']
 
# Set position of bar on X axis 
br1 = np.arange(len(IT)) 
br2 = [x + barWidth for x in br1] 

# Make the plot
plt.bar(br1, IT, color ='green', width = barWidth, 
        edgecolor ='grey', label ='Quantum') 
plt.bar(br2, ECE, color ='blue', width = barWidth, 
        edgecolor ='grey', label ='Classical')  
 
# Adding Xticks 
plt.xlabel('Simulations', fontweight ='bold', fontsize = 15) 
plt.ylabel('Exception Solutions', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(IT))], list(quantum['Disrupted File Inventory ID']), rotation = 90 )
 
plt.legend()
plt.show()