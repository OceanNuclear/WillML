import numpy as np
import sys
import pandas as pd
import pickle

MERGE_UNSCRAMBLE = True

try:
    import os
    os.chdir(sys.argv[1])
    if len(sys.argv[2:])>0:
        exec(''.join(sys.argv[2:]))
except IndexError:
    print("usage:")
    print("'python "+sys.argv[0]+" <folder_containing_preprocessed_data>/'")
    exit()
try:
    trlab = pd.read_pickle('trlab.pkl')
    prlab = pd.read_pickle('prlab.pkl')
except FileNotFoundError:
    print("Please preprocess that directory with 'MachineLearning.py' first.")
    print("Alternatively, if you're doing a debug run, make a copy of 'telab.pkl' and name it 'prlab.pkl'.")
    exit()

def get_category_conversion():
    with open('inversion_category.pkl', "rb") as f:
        return pickle.load(f)

def get_unscrambler():
    with open("order_original.pkl", "rb") as f:
        unscrambler = pickle.load(f)
    return unscrambler

def get_max_likelihood(df, conversion_dict):
    array = df.values #get the numpy array instead
    C1, l1, C2, l2 = [], [], [], []
    for row in array:
        if sum(row) !=0:
            cat = np.argsort(row)[::-1]
            likelihood = np.sort(row)[::-1]
            C1.append(conversion_dict[cat[0]])
            l1.append(likelihood[0])
            C2.append(conversion_dict[cat[1]])
            l2.append(likelihood[1])
        else:
            C1.append('N/A')
            l1.append(0)
            C2.append('N/A')
            l2.append(0)
    return pd.DataFrame({'1st':C1,'likelihood-1':l1,'2nd':C2,'likelihood-2':l2}, index = df.index)

conversion_dict = get_category_conversion()
pred = get_max_likelihood(prlab, conversion_dict)
train= get_max_likelihood(trlab, conversion_dict)
print("Please be patient, this appending is going to take a while...")

if MERGE_UNSCRAMBLE:
    result = pred.drop(pred.index) #create an empty dataframe with the correct titles.
    unscrambler_dict = get_unscrambler()
    indices = [ unscrambler_dict[i] for i in range(len(prlab)+len(trlab))]
    for i in indices:
        if i in prlab.index:
            result = result.append(pred.loc[i])
        else:
            assert i in trlab.index, "the indices are most likely incorrect."
            result = result.append(train.loc[i])
else:
    result = pred

result.to_csv("output.csv", header=None)