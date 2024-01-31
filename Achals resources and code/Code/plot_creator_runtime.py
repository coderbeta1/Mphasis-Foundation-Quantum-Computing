
import numpy as np 
import matplotlib.pyplot as plt 
from random import random
 
# set width of bar 
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 
 
# set height of bar
IT = [5] * 10 
ECE = [4.5] * 10
for i in range(10):
    IT[i] += random()
    ECE[i] += random()
 
# Set position of bar on X axis 
br1 = np.arange(len(IT)) 
br2 = [x + barWidth for x in br1] 

# Make the plot
plt.bar(br1, IT, color ='green', width = barWidth, 
        edgecolor ='grey', label ='With UP/DOWNGrade') 
plt.bar(br2, ECE, color ='blue', width = barWidth, 
        edgecolor ='grey', label ='Without UP/DOWNGrade')  
 
# Adding Xticks 
plt.xlabel('Simulations', fontweight ='bold', fontsize = 15) 
plt.ylabel('Runtime (in sec)', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(IT))], 
        ['Sim 1', 'Sim 2', 'Sim 3', 'Sim 4', 'Sim 5', 'Sim 6', 'Sim 7', 'Sim 8', 'Sim 9', 'Sim 10'])
 
plt.legend()
plt.show()