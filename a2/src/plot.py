import math as m
import numpy as np
import matplotlib.pyplot as plt

# Loading the data
# data = np.loadtxt('stream_triad_o3_uppmax.csv')
# bs1 = np.loadtxt('stream_triad_cuda_mupd_double_block_size_1.csv')
# bs128 = np.loadtxt('stream_triad_cuda_mupd_double_block_size_128.csv')
# bs256 = np.loadtxt('stream_triad_cuda_mupd_double_block_size_256.csv')
# bs512 = np.loadtxt('stream_triad_cuda_mupd_double_block_size_512.csv')
# bs1024 = np.loadtxt('stream_triad_cuda_mupd_double_block_size_1024.csv')
task1 = np.loadtxt('task1.csv')

# u = data[:]
# s = []
# t = []
# for j in range(len(u)):
#     s.append(u[j][1])
#     t.append(u[j][0])

# u1 = data1[:]
# s1 = []
# t1 = []
# u2 = data2[:]
# s2 = []
# t2 = []
# for j in range(len(u1)):
#     s1.append(u1[j][1])
#     t1.append(u1[j][0])
#     s2.append(u2[j][1])
#     t2.append(u2[j][0])

# u1 = bs1[:]
# u128 = bs128[:]
# u256 = bs256[:]
# u512 = bs512[:]
# u1024 = bs1024[:]
u = task1[:]
# s1 = []
# s128 = []
# s256 = []
# s512 = []
# s1024 = []
s = []
# t1 = []
# t128 = []
# t256 = []
# t512 = []
# t1024 = []
t = []

# for j in range(len(u128)):
#     s1.append(u1[j][1])
#     t1.append(u1[j][0])
#     s128.append(u128[j][1])
#     t128.append(u128[j][0])
#     s256.append(u256[j][1])
#     t256.append(u256[j][0])
#     s512.append(u512[j][1])
#     t512.append(u512[j][0])
#     s1024.append(u1024[j][1])
#     t1024.append(u1024[j][0])
for j in range(len(u)):
    s.append(u[j][1])
    t.append(u[j][0])

# Plotting the data
plt.semilogx(t, s, label = "Initial")
# plt.semilogx(t1, s1, label = "Block size 1")
# plt.semilogx(t128, s128, label = "Block size 128")
# plt.semilogx(t256, s256, label = "Block size 256")
# plt.semilogx(t512, s512, label = "Block size 512")
# plt.semilogx(t1024, s1024, label = "Block size 1024")
# plt.semilogx(t, s, label = "Cluster with O3")
# plt.semilogx(t1, s1, label = "Local with align")
# plt.semilogx(t2, s2, label = "Local without align")
plt.legend()
plt.xlabel("Size")
plt.ylabel("GB/s")
plt.title("Throughput")

plt.grid()
plt.show()
exit()
