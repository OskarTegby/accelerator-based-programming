import math as m
import numpy as np
import matplotlib.pyplot as plt

# Loading the data
native_sq = np.loadtxt('native_square.csv')
native_r1 = np.loadtxt('native_rect1.csv')
native_r2 = np.loadtxt('native_rect2.csv')

cublas_sq = np.loadtxt('cublas_square.csv')
cublas_r1 = np.loadtxt('cublas_rect1.csv')
cublas_r2 = np.loadtxt('cublas_rect2.csv')

u1_sq = native_sq[:]
u1_r1 = native_r1[:]
u1_r2 = native_r2[:]

u2_sq = cublas_sq[:]
u2_r1 = cublas_r1[:]
u2_r2 = cublas_r2[:]

s1_sq = []
s1_r1 = []
s1_r2 = []

s2_sq = []
s2_r1 = []
s2_r2 = []

t1_sq = []
t1_r1 = []
t1_r2 = []

t2_sq = []
t2_r1 = []
t2_r2 = []

for j in range(len(u1_sq)):
    s1_sq.append(u1_sq[j][1])
    s2_sq.append(u2_sq[j][1])
    t1_sq.append(u1_sq[j][0])
    t2_sq.append(u2_sq[j][0])

for j in range(len(u2_r1)):
    s1_r1.append(u1_r1[j][1])
    s2_r1.append(u2_r1[j][1])
    t1_r1.append(u1_r1[j][0])
    t2_r1.append(u2_r1[j][0])
    
for j in range(len(u1_r2)):
    s1_r2.append(u1_r2[j][1])
    s2_r2.append(u2_r2[j][1])
    t1_r2.append(u1_r2[j][0])
    t2_r2.append(u2_r2[j][0])

# Plotting the data
plt.figure(0)
plt.loglog(t1_sq, s1_sq)
plt.grid()
plt.xlabel("Size")
plt.ylabel("GB/s")
plt.title("Throughput, Case A (Native)")

plt.figure(1)
plt.plot(t1_sq, s1_sq)
plt.grid()
plt.xlabel("Size")
plt.ylabel("GB/s")
plt.title("Throughput, Case A (Native)")

plt.figure(2)
plt.loglog(t2_sq, s2_sq)
plt.grid()
plt.xlabel("Size")
plt.ylabel("GB/s")
plt.title("Throughput, Case A (cuBLAS)")

plt.figure(3)
plt.plot(t2_sq, s2_sq)
plt.grid()
plt.xlabel("Size")
plt.ylabel("GB/s")
plt.title("Throughput, Case A (cuBLAS)")

plt.figure(4)
plt.plot(t1_r1, s1_r1)
plt.grid()
plt.xlabel("Size")
plt.ylabel("GB/s")
plt.title("Throughput, Case B (Native)")

plt.figure(5)
plt.plot(t2_r1, s2_r1)
plt.grid()
plt.xlabel("Size")
plt.ylabel("GB/s")
plt.title("Throughput, Case B (cuBLAS)")

plt.figure(6)
plt.plot(t1_r2, s1_r2)
plt.grid()
plt.xlabel("Size")
plt.ylabel("GB/s")
plt.title("Throughput, Case C (Native)")

plt.figure(7)
plt.plot(t2_r2, s2_r2)
plt.grid()
plt.xlabel("Size")
plt.ylabel("GB/s")
plt.title("Throughput, Case C (cuBLAS)")

plt.show()
exit()
