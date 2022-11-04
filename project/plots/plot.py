import math as m
import numpy as np
import matplotlib.pyplot as plt

# Loading the data
cpu_double_gbs = np.loadtxt('csv/cpu_double_gbs.csv')
cpu_float_gbs = np.loadtxt('csv/cpu_float_gbs.csv')
cpu_double_mupds = np.loadtxt('csv/cpu_double_mupds.csv')
cpu_float_mupds = np.loadtxt('csv/cpu_float_mupds.csv')
gpu_double_gbs = np.loadtxt('csv/gpu_double_mupds.csv')
gpu_float_gbs = np.loadtxt('csv/gpu_float_mupds.csv')
gpu_double_mupds = np.loadtxt('csv/gpu_double_gbs.csv')
gpu_float_mupds = np.loadtxt('csv/gpu_float_gbs.csv')

u1 = cpu_double_gbs[:]
u2 = cpu_float_gbs[:]
u3 = cpu_double_mupds[:]
u4 = cpu_float_mupds[:]
u5 = gpu_double_gbs[:]
u6 = gpu_float_gbs[:]
u7 = gpu_double_mupds[:]
u8 = gpu_float_mupds[:]
s1 = []
t1 = []
s2 = []
t2 = []
s3 = []
t3 = []
s4 = []
t4 = []
s5 = []
t5 = []
s6 = []
t6 = []
s7 = []
t7 = []
s8 = []
t8 = []

for j in range(len(u1)):
    s1.append(u1[j][1])
    t1.append(u1[j][0])

    s2.append(u2[j][1])
    t2.append(u2[j][0])
    
    s3.append(u3[j][1])
    t3.append(u3[j][0])
    
    s4.append(u4[j][1])
    t4.append(u4[j][0])
    
    s5.append(u5[j][1])
    t5.append(u5[j][0])
    
    s6.append(u6[j][1])
    t6.append(u6[j][0])
    
    s7.append(u7[j][1])
    t7.append(u7[j][0])
    
    s8.append(u8[j][1])
    t8.append(u8[j][0])

# Plotting the data
# plt.plot(t1, s1, label = "cpu_double_gbs")
# plt.plot(t2, s2, label = "cpu_float_gbs")
# plt.plot(t3, s3, label = "cpu_double_mupds")
# plt.plot(t4, s4, label = "cpu_float_mupds")
plt.plot(t5, s5, label = "gpu_double_gbs")
plt.plot(t6, s6, label = "gpu_float_gbs")
# plt.plot(t7, s7, label = "gpu_double_mupds")
# plt.plot(t8, s8, label = "gpu_float_mupds")
plt.legend()
plt.xlabel("Size")
plt.ylabel("MUPD/s")
plt.title("Throughput")

plt.grid()
plt.show()
exit()
