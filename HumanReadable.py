import numpy as np
import pandas as pd
import pickle

UNLABELLED = -1
LOGSVM = False
MERGE_UNSCRAMBLE = True

try:
    import os, sys
    os.chdir(sys.argv[1])
    if len(sys.argv[2:])>0:
        exec(''.join(sys.argv[2:]))
except IndexError:
    print("usage:")
    print("'python "+sys.argv[0]+" <folder_containing_preprocessed_data>/'")
    print("add 'MERGE_UNSCRAMBLE = False' immediately after that in here ^ if you did not shuffle the dataframe before prediction")
    exit()
try:
    trlab = pd.read_pickle('trlab.pkl')
    LINEARSVM = not LOGSVM
    if LINEARSVM:
        linsvm_pred = pd.read_pickle('linsvm_pred.pkl')
    elif LOGSVM:
        logsvm_pred = pd.read_pickle('logsvm_pred.pkl')
    if not os.path.isfile("order_original.pkl"):
        MERGE_UNSCRAMBLE = False # if no order_original.pkl exist, that means the Shuffle.py step was skipped, thus there is no need to unscramble.
except FileNotFoundError:
    print("Please preprocess that directory with 'MachineLearning.py' first.")
    print("Alternatively, if you're doing a debug run, make a copy of 'telab.pkl' and name it 'linsvm_pred.pkl'.")
    exit()


def get_category_conversion():
    with open('inversion_category.pkl', "rb") as f:
        return pickle.load(f)

def get_unscrambler():
    with open("order_original.pkl", "rb") as f:
        unscrambler = pickle.load(f)
    return unscrambler

def get_max_likelihood(df, inversion_dict):
    array = df.values #get the numpy array instead
    C1, l1, C2, l2 = [], [], [], []
    if array.ndim>1:
        for row in array:
            cat = np.argsort(row)[::-1]
            likelihood = np.sort(row)[::-1]
            C1.append(inversion_dict[cat[0]])
            l1.append(likelihood[0])
            C2.append(inversion_dict[cat[1]])
            l2.append(likelihood[1])
    else:
        for cat_int in array:
            if cat_int!=UNLABELLED:
                C1.append(inversion_dict[cat_int]) #convert the sparse embedding (integer) representation of the category back to text.
            else:
                C1.append('UNLABELLED') #This should never happen as we expect all of the training set to have been labelled.
            l1.append(1.0)
            C2.append('N/A')
            l2.append(0)
    return pd.DataFrame({'1st':C1,'distance-1':l1,'2nd':C2,'distance-2':l2}, index = df.index)

inversion_dict = get_category_conversion()
if LINEARSVM:
    pred = get_max_likelihood(linsvm_pred, inversion_dict)
    print("Finished getting the distance to hyperplane for {0}, now doing the same for {1}".format("linsvm_pred", "the training set"))
else:
    pred = get_max_likelihood(logsvm_pred, inversion_dict)
    print("Finished getting the distance to hyperplane for {0}, now doing the same for {1}".format("logsvm_pred", "the training set"))
train= get_max_likelihood(trlab, inversion_dict)
print("Finished getting the distance to hyperplane for the training set.")

if MERGE_UNSCRAMBLE:
    print("Now composing the final, unscrambled sheet...")
    result = train.drop(train.index) #create an empty dataframe with the correct titles.
    unscrambler_dict = get_unscrambler()
    indices = [ unscrambler_dict[i] for i in range(len(linsvm_pred)+len(trlab))]
    for i in indices:
        if i in linsvm_pred.index:
            result = result.append(pred.loc[i])
        else:
            assert i in trlab.index, "The index does not exist!"
            result = result.append(train.loc[i])
else:
    result = pred

result.to_csv("output.csv", header=None)