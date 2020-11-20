# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from pg import BertPolicyGradientAlgo, NoBertPolicyGradientAlgo
from pg import Lang

def loadData():
    df = pd.read_csv("data/sighan10.csv")
    dataset = df.to_numpy()
    np.random.shuffle(dataset)
    split_index = int(len(dataset)*0.9)
    training_data, test_data = dataset[:split_index, :], dataset[split_index:, :]
    return training_data, test_data


import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
    
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    #loc = ticker.MultipleLocator(base=0.2)
    #ax.yaxis.set_major_locator(loc)
    for (l,p) in points.items():
        plt.plot(p, label=l)
    plt.legend()
    plt.show()



if __name__ == "__main__":
    training_data, test_data = loadData()
    n_iter = 500
    #algo = BertPolicyGradientAlgo(n_iter, 10, training_data)
    algo = NoBertPolicyGradientAlgo(n_iter, 10, training_data, test_data)
    algo.run()
    #fig = plt.figure()
    #plt.plot([i for i in range(n_iter)], np.array(algo.rewards)[:,0,0])
    #plt.show()
    showPlot({"Train": algo.plot_losses, "Eval": algo.eval_losses})

    # torch.Size([1, 1]) torch.Size([1, 1, 768]) torch.Size([1, 52, 768])
    # torch.Size([1, 1, 768])

    # torch.Size([1, 1]) torch.Size([1, 1, 256]) torch.Size([52, 256])
    # torch.Size([1, 1, 256])