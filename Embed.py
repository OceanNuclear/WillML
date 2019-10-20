from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
import pandas as pd
import pickle

UNLABELLED = -1
try:
    import os, sys
    os.chdir(sys.argv[1])
    if len(sys.argv[2:])>0:
        exec(''.join(sys.argv[2:]))
except IndexError:
    print("usage:")
    print("'python "+sys.argv[0]+" <folder_containing_RawData.csv>/'")
    sys.exit()
try:
    tr_frame = pd.read_csv("train_data.csv", header=None, index_col=0).fillna('')
    te_frame = pd.read_csv("test_data.csv", header=None, index_col=0).fillna('')
except FileNotFoundError:
    print("Please preprocess that directory with 'Shuffle.py' first;")
    print("Or, alternatively, save the 'train_data.csv' and 'test_data.csv' directly.")
    exit()

def get_unique_words(descriptions):
    allwords = " ".join(list(descriptions))   #join them into a single string
    allwords = allwords.lower().split(" ") #change all words to lower case to minimize duplications
    uniquewords = list(set(allwords))  #turn it into an iterable set
    uniquewords.sort()
    uniquewords = [ i for i in uniquewords if (i!="" and i.isprintable())] #remove the empty string character from the set
    print("{0} unique words is found and used as the features for learning.".format(len(uniquewords)))
    return uniquewords #a list of all unique words

def save_inversion_dict(set_as_list, data="category"):
    allcombo_dict = dict([ [ i,set_as_list[i] ] for i in range(len(set_as_list))]) #convert to a dictionary, which can be saved.
    with open("inversion_"+data+".pkl","bw") as f:
        pickle.dump(allcombo_dict, f)
    return

def save_conversion_dict(set_as_list, data="category"):
    allcombo_dict = dict([ [set_as_list[i], i]  for i in range(len(set_as_list)) ])
    with open("conversion_"+data+".pkl","bw") as f:
        pickle.dump(allcombo_dict, f)
    return    

def get_set_of_categroies(pdseries):
    s = set(pdseries)
    s.discard('')
    list_of_set_of_category = list(s)
    list_of_set_of_category.sort()
    return list_of_set_of_category

def convert2embedding(pdseries, uniquewords_list):
    new_series = {}
    for index,row in pdseries.iteritems():
        row=row.lower().split(" ")
        #use number of occurance instead of boolean
        new_series.update({index : [row.count(word) for word in uniquewords_list]})
    return pd.Series(new_series)

def convert2integerrepresentation(pdseries, category_list):
    new_series = {}
    for index,row in pdseries.iteritems():
        #no lower() operation needed
        try:
            integer_requried = category_list.index(row)#find the matching value
        except ValueError:
            integer_requried = UNLABELLED
        new_series.update({index:integer_requried})
    return pd.Series(new_series)

def convert2onehot(pdseries):
    s = set(pdseries)
    s.discard(UNLABELLED)
    all_categories = list(s) # using the integer representation, stored in 'category_conversion.pkl', as the column names.
    all_categories.sort()
    series = list(pdseries)
    onehot_result = np.zeros((len(series),len(all_categories)))
    for i in range(len(series)):
        try:
            onehot_result[i][all_categories.index(series[i])]+=1
        except ValueError: 
            assert series[i]==UNLABELLED, "This entry has not been labelled with a member of the set of accepted categrories; but it isn't labelled with the integer representation -1 to indicate that it was UNLABELLED to begin with either; so something is wrong."
    onehot_result = pd.DataFrame(onehot_result, index=pdseries.index, columns=all_categories)
    return onehot_result #return a DataFrame

def split2columns(pdseries):
    assert type(pdseries) == pd.core.series.Series, "only accept a single column (pd.Series) object"
    new_df = []
    for vector in pdseries:
        new_df.append(vector)
    new_df = pd.DataFrame(new_df, index = pdseries.index, columns = [ i for i in range(len(vector)) ])
    return new_df #return another DataFrame

#do the embedding conversion
hybrid_data = tr_frame[1]+" "+ tr_frame[2]#combine column 1 (description) and 2 (class)
uniquewords = get_unique_words(hybrid_data) #combine the "description" with "class" to get the hybrid embedding

save_inversion_dict(uniquewords, data="description_and_class")
save_conversion_dict(uniquewords, data="description_and_class")
list_of_set_of_category = get_set_of_categroies(tr_frame[3])
save_inversion_dict(list_of_set_of_category)
save_conversion_dict(list_of_set_of_category)

#save dataframe with only embedding
frames = {'tr':tr_frame, 'te':te_frame}
train_test_feature_label = {}

for t in ('tr', 'te'):
    embedding_representation = convert2embedding(hybrid_data, uniquewords)
    category_integer_representation = convert2integerrepresentation(frames[t][3], list_of_set_of_category)
    # preprocessed_data = pd.DataFrame({'feature':embedding_representation,'label':category_integer_representation})
    train_test_feature_label.update({t+'fea':split2columns(embedding_representation)})
    train_test_feature_label.update({t+'lab':convert2onehot(category_integer_representation)})

#   train_test_feature_label = {'trfea': ,
                                # 'trlab': ,
                                # 'tefea': ,
                                # 'telab': ,
                                # }

for k,v in train_test_feature_label.items():
    with open(k+'.pkl', 'wb') as f:
        pickle.dump(v,f)