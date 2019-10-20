import numpy as np
import pandas as pd
import time
from shutil import get_terminal_size

#cd into the directory to be operated on.
try:
    import os, sys
    os.chdir(sys.argv[1])
    if len(sys.argv[2:])>0:
        exec(''.join(sys.argv[2:]))
except IndexError:
    print("usage:")
    print("'python "+sys.argv[0]+" <folder_containing_preprocessed_data>/'")
    exit()
#Read the training, testing, etc. data.
try:
    trfea = pd.read_pickle("trfea.pkl")
    trlab = pd.read_pickle("trlab.pkl")
    tefea = pd.read_pickle("tefea.pkl")
    telab = pd.read_pickle("telab.pkl")
except FileNotFoundError:
    print("Please cut up the data with 'Embed.py' first.")
    exit()

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score

random_forest = RandomForestClassifier(n_estimators=200, max_depth=3),
svm           = LinearSVC(),
naive_bayes   = MultinomialNB(),
logreg        = LogisticRegression(random_state = 0),

'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations
tf.random.set_random_seed(0)

# neural_network_structure = []
hyperparamteres             ={
                             'activaiton function' : activations.softmax,
                             'num_epochs': 10,
                             'learning_rate' :0.0001,
                             # 'loss_func' : 'binary_crossentropy', #tf.keras.losses.
                             'loss_func' : 'categorical_crossentropy',
                             'metrics': ['Accuracy'],
                             'regularize': True} #tf.keras.metrics.
print(trfea)
# mlresult = convert2onehot(sheet['label'])
# mlresult.to_pickle("mlresult.pkl")
'''