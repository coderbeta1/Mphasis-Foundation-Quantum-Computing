
import numpy as np 
import matplotlib.pyplot as plt 
 
# set width of bar 
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 
 
# set height of bar 
IT = [100] * 10 
ECE = [100] * 10
ECE[3] = 97
ECE[9] = 99
ECE[4] = 98
ECE[2] = 99
 
# Set position of bar on X axis 
br1 = np.arange(len(IT)) 
br2 = [x + barWidth for x in br1] 

# Make the plot
plt.bar(br1, IT, color ='red', width = barWidth, 
        edgecolor ='grey', label ='With UP/DOWNGrade') 
plt.bar(br2, ECE, color ='orange', width = barWidth, 
        edgecolor ='grey', label ='Without UP/DOWNGrade')  
 
# Adding Xticks 
plt.xlabel('Simulations', fontweight ='bold', fontsize = 15) 
plt.ylabel('Number or PNRs Accomodated (in %)', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(IT))], 
        ['Sim 1', 'Sim 2', 'Sim 3', 'Sim 4', 'Sim 5', 'Sim 6', 'Sim 7', 'Sim 8', 'Sim 9', 'Sim 10'])
 
plt.legend()
plt.show()