#!/home/ocean/anaconda3/bin/python3
from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
import pandas as pd
import pickle

SHUFFLE_SEED = 0
TRAIN_THRES = 1
try:
    import os, sys
    os.chdir(sys.argv[1])
except IndexError:
    print("usage:")
    print("'python Embedding.py <folder_containing_RawData.csv>/'")
    exit()

def shuffle(df, category_col=3):
    #record the original order of the indices
    original_indices = df.index
    original_order = dict([(original_indices[i], i) for i in range(len(df)) ]) #a dictionary to output iloc when loc is inputted.

    #shuffle it once.
    np.random.seed(SHUFFLE_SEED)
    indices = df.index.tolist()
    np.random.shuffle(indices)
    df_first_shuffle = df.loc[indices]

    #sort them into len(set(df[category_col])) piles.
    sorted_by_category_then_by_iloc_in_shuffled_order = []
    for i in list(set(df[category_col])):
        matching_in_shuffled_list = df_first_shuffle[ df_first_shuffle[category_col]==i ]
        matching_indices_in_shuffled_order = matching_in_shuffled_list.index.tolist()
        sorted_by_category_then_by_iloc_in_shuffled_order.append(matching_indices_in_shuffled_order)
    
    #Fill up to the TRAIN_THRES, and then fill in the rest.
    within_threshold = []
    for short_list in sorted_by_category_then_by_iloc_in_shuffled_order:
        within_threshold+=short_list[:TRAIN_THRES]    #within training threshold
    beyond_threshold = []
    for i in df_first_shuffle.index:
        if not (i in within_threshold):
            beyond_threshold.append(i)
    final_order = within_threshold + beyond_threshold

    #save the conversion required.
    unscramble = [(i,original_order[final_order[i]]) for i in range(len(sheet))] #a dictionary storing the .iloc index (after, before)
    with open('unscramble.pkl', 'wb') as f:
        pickle.dump(dict(unscramble), f)
    df_final = df.loc[final_order]
    return df_final

def get_unique_words(descriptions):
    allwords = " ".join(list(descriptions))   #join them into a single string
    allwords = allwords.lower().split(" ") #change all words to lower case to minimize duplications
    uniquewords = list(set(allwords))  #turn it into an iterable set
    uniquewords = [ i for i in uniquewords if (i!="" and i.isprintable())] #remove the empty string character from the set
    print("{0} unique words is found and used as the features for learning.".format(len(uniquewords)))
    return uniquewords #a list of all unique words

def get_set_of_categroies(pdseries):
    list_of_set_of_category = list(set(pdseries))
    return list_of_set_of_category

def save_as_dict(set_as_list, data="category"):
    allcombo_dict = dict([ [ i,set_as_list[i] ] for i in range(len(set_as_list))]) #convert to a dictionary, which can be saved.
    with open(data+"_conversion.pkl","bw") as f:
        pickle.dump(allcombo_dict, f)
    return

def convert2embedding(pdseries, uniquewords_list):
    new_series = {}
    for index,row in pdseries.iteritems():
        row=row.lower()
        new_series.update({index : [(word in row) for word in uniquewords_list]})
    return pd.Series(new_series)

def convert2integerrepresentation(pdseries, category_list):
    new_series = {}
    for index,row in pdseries.iteritems():
        #no lower() operation needed
        integer_requried = category_list.index(row)#find the matching value
        new_series.update({index:integer_requried})
    return pd.Series(new_series)

def get_dict(file):
    with open(file, "rb") as f:
        return pickle.load(f)

def convertback2words(pdseries, uniquewords_dict):
    pass

#read
sheet = pd.read_csv("RawData.csv", header=None, index_col = 0) #header=['index', 'description', 'class', 'category']

#Shuffle
if SHUFFLE_SEED!=None:
    sheet = shuffle(sheet)

#do the embedding conversion
hybrid_data = sheet[1].fillna('')+" "+ sheet[2].fillna('')#combine column 1 (description) and 2 (class)
uniquewords = get_unique_words(hybrid_data) #combine the "description" with "class" to get the hybrid embedding
save_as_dict(uniquewords, data="description_and_class")
list_of_set_of_category = get_set_of_categroies(sheet[3])
save_as_dict(list_of_set_of_category)


#save dataframe with only embedding
embedding_representation = convert2embedding(hybrid_data, uniquewords)
category_integer_representation = convert2integerrepresentation(sheet[3], list_of_set_of_category)
preprocessed_data = pd.DataFrame({'feature':embedding_representation,'label':category_integer_representation})
preprocessed_data.to_csv('preprocessed_data.csv')