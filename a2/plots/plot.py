import math as m
import numpy as np
import matplotlib.pyplot as plt

# Loading the data
task1 = np.loadtxt('task1.csv')
u = task1[:]
s = []
t = []

for j in range(len(u)):
    s.append(u[j][1])
    t.append(u[j][0])

# Plotting the data
plt.plot(t, s, label = "Initial")
plt.legend()
plt.xlabel("Size")
plt.ylabel("GB/s")
plt.title("Throughput")

plt.grid()
plt.show()
exit()
