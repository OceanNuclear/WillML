from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
import pandas as pd
import pickle

TRAIN_FRAC = 0.5
SHUFFLE_SEED = 0
TRAIN_THRES = 1
UNLABELLED = -1

try:
    import os, sys
    os.chdir(sys.argv[1])
    if len(sys.argv[2:])>0:
        exec(''.join(sys.argv[2:]))
except IndexError:
    print("usage:")
    print("'python "+sys.argv[0]+" <folder_containing_RawData.csv>/'")
    exit()

def shuffle(df, category_col=3):
    #record the original order of the indices
    original_indices = df.index
    original_order = dict([(i, original_indices[i]) for i in range(len(df)) ]) #a dictionary to output iloc when loc is inputted.
    with open('order_original.pkl', 'wb') as f:
        pickle.dump(original_order, f)

    #shuffle it once.
    np.random.seed(SHUFFLE_SEED)
    indices = df.index.tolist()
    np.random.shuffle(indices)
    df_first_shuffle = df.loc[indices]

    #sort them into len(set(df[category_col])) piles.
    sorted_by_category_then_by_iloc_in_shuffled_order = []
    s = set(df[category_col])
    s.discard('')
    l = list(s)
    l.sort() #N.B. must sort the list after converting it from the set, as sets are unordered, and changes everytime.
    for i in l:
        matching_in_shuffled_list = df_first_shuffle[ df_first_shuffle[category_col]==i ]
        matching_indices_in_shuffled_order = matching_in_shuffled_list.index.tolist()
        sorted_by_category_then_by_iloc_in_shuffled_order.append(matching_indices_in_shuffled_order)
    
    #sort them into training and testing set under some constraints.
    within_threshold, beyond_threshold, unlabelled = [], [], []
    #Fill up to the TRAIN_THRES
    for short_list in sorted_by_category_then_by_iloc_in_shuffled_order:
        within_threshold+=short_list[:TRAIN_THRES]    #within training threshold
    #file away unlabelled data to the end
    unlabelled = df_first_shuffle[df_first_shuffle[category_col]==''].index.tolist()
    for i in df_first_shuffle.index:
        if not ((i in within_threshold) or (i in unlabelled)):
            beyond_threshold.append(i)
    final_indices = within_threshold + beyond_threshold + unlabelled

    #save the conversion required.
    final_order = dict([(i, final_indices[i]) for i in range(len(df))])
    with open('order_final.pkl', 'wb') as f:
        pickle.dump(final_order, f)

    inverse_final_indices = dict([(final_indices[i], i) for i in range(len(df))])
    iloc_new_to_iloc_old = dict([(i, inverse_final_indices[original_order[i]]) for i in range(len(df))])
    with open('order_iloc_new_to_iloc_old.pkl', 'wb') as f:
        pickle.dump(iloc_new_to_iloc_old, f)
    df_final = df.loc[final_indices]
    return df_final

#read
sheet = pd.read_csv("RawData.csv", header=None, index_col = 0).fillna('') #header=['index', 'description', 'class', 'category']

#Shuffle
if SHUFFLE_SEED!=None:
    sheet = shuffle(sheet) #saves the ordering before and after.
num_train = int(np.round(len(sheet)*TRAIN_FRAC))
num_test = len(sheet) - num_train

tr_frame = sheet[:num_train]
te_frame = sheet[num_train:]

#Split into training and testing set
tr_frame.to_csv("train_data.csv", header=None)
te_frame.to_csv("test_data.csv", header=None)