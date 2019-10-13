#!/home/ocean/anaconda3/bin/python3
from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
import pandas as pd

#TRAIN_FRAC = 0.5

#catch errors
try:
    import os, sys
    os.chdir(sys.argv[1])
except IndexError:
    print("usage:")
    print("'python MachineLearning.py <folder_containing_preprocessed_data>/'")
    exit()
try:
    pd.read_csv("preprocessed_data.csv", index_col=0)
except FileNotFoundError:
    print("Please preprocess that directory with 'Embed.py' first.")
    exit()

#do the machine learning

# #split into training and testing set
# num_train = int(np.round(len(sheet)*TRAIN_FRAC))
# num_test = len(sheet) - num_train