import torch
#import torchvision
import sys
import tkinter as tk
from envs.maze import Maze
import matplotlib.pyplot as plt
import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# print(torch.cuda.is_available())



# print(torch.__version__)
# print(torch.version.cuda)

#print(torchvision.__version__)
#print(torchvision.version.cuda)


# print(sys.version_info.major)



def plot_cost():
    plt.plot(np.arange(len(cost_his)), cost_his)
    plt.ylabel('Cost')
    plt.xlabel('training steps')
    plt.show()


cost_his = []
cost_his.append(1)
cost_his.append(2)
cost_his.append(3)
cost_his.append(4)

plot_cost()