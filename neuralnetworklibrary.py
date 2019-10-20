import glob
import sys
import os
# Import commonly used numerical processing and plotting functions
import pandas as pd
import matplotlib as mpl
mpl.use("agg") #for using this script on the cumulus server of ukaea
from matplotlib import pyplot as plt
import numpy as np
from numpy import e
from numpy.fft import fft, ifft
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations
import time
from shutil import get_terminal_size
import fcntl

def read_NN_weights(session_name):
    '''returns the weights and biases read from a h5 file'''
    import h5py
    path_to_file = ".checkpoints/" + session_name.split("/")[-1] + ".h5"
    weights, biases = {}, {}
    keys = []
    with h5py.File(path_to_file, 'r') as f:  # open file
        f.visit(keys.append)  # append all keys to list by visiting each
        for k in keys:
            if ':' in k:  # Filter out all keys that is
                # Get the layer number
                name_splitted = f[k].name.split("/")
                layer = name_splitted[1]

                layer_num = "".join(d for d in layer if d.isdigit())
                if layer_num == "": layer_num = "1"  # if there is no digit in the layer name: it must've been layer 1.

                # Decide whether it's bias or weight according to the last element in the name_splitted list
                if "kernel" in name_splitted[-1]:
                    weights["layer_" + layer_num] = f[k].value
                elif "bias" in name_splitted[-1]:
                    biases["layer_" + layer_num] = f[k].value
    return weights, biases

def _find_matching_braces(list_of_lines):
    '''given a collection of text lines stored as a list, find out the indices of the lines where matching braces occurs'''
    # copying the design pattern of finding matching paranthesis.
    brace_stack = []  # stack
    d = {}
    # d stores the opening and closing braces' line numbers
    for l_num, line in enumerate(list_of_lines):
        if "{" in line: brace_stack.append(l_num)
        if "}" in line:
            try:
                d[brace_stack.pop()] = l_num
            except IndexError:
                print("More } than {")
    if len(brace_stack) != 0: print("More { than }")
    return d

def convert_str_value(string):
    if ("[" in string) and ("]" in string):  # filter out the list
        splitted_list = string.strip("[]").split(",")
        # filter out the empty list case:
        if len(splitted_list) == 1:
            if splitted_list[0] == "":
                # return an empty list [] instead of a [None]
                return []
        return_list = [convert_str_value(elem.strip()) for elem in
                       splitted_list]  # recursively call itself on the elements of the list
        return return_list
    if string.startswith('"') and string.endswith('"'):  # filter out the strings
        assert string.count('"') == 2, "too many quotation marks!"
        return string[1:-1]
    if string.startswith("'") and string.endswith("'"):  # filter out the strings
        assert string.count("'") == 2, "too many quotation marks!"
        return string[1:-1]
    if "False" in string: return False  # filter out the booleans and None's
    if "True" in string: return True
    if "None" in string: return None
    if ("." in string) or ("e-0" in string):  # filter out the floats
        try:
            return float(string)
        except ValueError:  # filter out the function objects
            if ("<" in string) and (">" in string) and ("object" in string):
                raise ValueError("Cannot input a method object as a string; but can try using string e.g. 'AdaGrad'")
    return int(string)  # only integers should be left

def cut_file_in_halves(filename):
    '''
    return two lists, one containing the first dictionary;
    the other contains all other files.
    '''
    with open(filename, "r") as f:
        data = f.readlines()
    braces = _find_matching_braces(data)
    try:
        first_pair = next(iter(braces.items()))
    except StopIteration:
        sys.exit("No more dictionaries in file")
    first_dict = []
    for line in data[first_pair[0]:first_pair[1] + 1]:
        if ":" in line:
            line = line.strip().strip("{}").strip()
            if line[-1] == ",": line = line[:-1]  # remove the rightmost comma
            first_dict.append(line)
    rest_of_the_lines = data[first_pair[1] + 1:]
    return first_dict, rest_of_the_lines

def convert_lines_to_dict(lines):
    dictionary_to_be_returned = {}
    for line in lines:
        # split the "sentence" down the middle at the ':'
        key, value = [arg.strip() for arg in line.split(":")]
        # must ensure that none of these are empty
        assert not len(key) == 0, "Must have a key before the :"
        assert not len(value) == 0, "Must have a value after the :"
        dictionary_to_be_returned[key] = convert_str_value(value)
    return dictionary_to_be_returned

def overwrite_file_by_removing_first_dict(filename, lines):
    # fcntl.flock(filename, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with open(filename, "w") as f:
        for line in lines:
            f.write(line)
    # fcntl.flock(filename, fcntl.LOCK_UN)

def fold_and_append(response_matrix, label, log_label):
    if log_label: #exponentiate, multiply, and then take log again:
        label_in_linear = tf.math.pow(e,label)
        pred_feature_in_linear = tf.matmul(label_in_linear, response_matrix.T.astype("float32"))
        pred_feature = tf.math.log(pred_feature_in_linear)
    elif not log_label:
        pred_feature = tf.tensordot(response_matrix, label)
    return tf.concat([label, pred_feature], axis=1)

def convert_str_to_loss_func(string, response_matrix, log_label):
    include_folding_string = "_including_folded_reaction_rates"
    if string.endswith(include_folding_string):
        loss_func = convert_str_to_loss_func( string.replace(include_folding_string,""), response_matrix, log_label=log_label) #use to get one of the following
        return lambda lab, pred : loss_func( fold_and_append(response_matrix,lab, log_label=log_label), fold_and_append(response_matrix,pred, log_label=log_label)) #return a wrapped function
        #This assumes that each element of the folded reaction rate has the same weight in terms of deviation from the label. 
    elif string=="mean_squared_error":
        return tf.compat.v1.losses.mean_squared_error
    elif string=="mean_pairwise_squared_error":
        return tf.compat.v1.losses.mean_pairwise_squared_error
    elif string=="cosine_distance":
        return lambda lab,pred : tf.losses.cosine_distance(lab,pred,axis=0)
    else:
        return string


class NeuralNetwork():
    #This class contains all the read- and write information required to pre-process and post-process inputs to the neural network.
    #It allows for all types of input imaginable, except for the 
    def __init__(self):
        # List of parameters for pre- and post-processing
        self.data_preparation_options = {
                "log_feature"   : True,
                "log_label"     : True,
                "lower_limit"   : 1E-12, # any flux value below lower_limt will be clipped to lower_limt
                "label_already_in_PUL" : False, #labels needs to be converted
                #from total flux per bin to average flux per unit lethargy (PUL) across the bin before training/handled by the NN.
                #total flux needs to be divided by the difference in lethargy of the upper and lower limit to be converted into flux PUL.
                "ft_label"      : False, # Do not apply fourier transform before processing the data by default.
                # apply log on both sides (the RR and the flux) before processing the data, by default
        }
        
        # options of how to rearrange the data before reading it in.
        self.data_reordering_options = {
                "shuffle_seed"      : 0,
                "startoff"          : None,
                "cutoff"            : None, #number of data lines to accept from the next file.
                "train_split"       : 0.8,
                "validation_split"  : 0.2,
        }
        
        # metadata recording the training time. These will be auto-generated as the NeuralNetwork training begins.
        self.timing = {
                "start_time_raw"    : time.time(), #give in unix time
                "start_time"        : time.strftime("%I:%M%p %d-%m-%Y").lower(),
                "run_time_seconds"  : 0.0,
        }

        start_time_global = self.timing["start_time_raw"]
        
        # hyperparameter describing the architecture of the NN
        self.hyperparameter = {
            "tf_seed"       : 0,
            "act_func"      : [],
            "hidden_layer"  : [],
            "learning_rate" : 0.001,
            "loss_func"     : 
                            "mean_pairwise_squared_error",
                            # "cosine_distance", # chi^2 calculated as normalized unit sum of squared values #this one is weird and I can never get it to work.
                            # "mean_squared_error", #chi^2 calculated as mean of squares of deviation from true labels.
            "metrics"       : ['mean_absolute_error', 'mean_squared_error'], # "precision_at_thresholds" only works with boolean, therefore is not used.
            "num_epochs"    : 10000,
        }
        self.hyperparameter["optimizer"] = tf.keras.optimizers.Adam(self.hyperparameter["learning_rate"])

        #loss values to be filled in later
        self.losses = {
        }
        # for key in self.hyperparameter['metrics']:
        #     self.losses.update({key: 0})

        self.session_name = ""
        
        self.callbacks_applied = ["PrintEpochInfo"]#, "TensorBoard"]

        # list the parameters to be saved
        self.settable_property_list = list(self.__dict__.keys())
        
        ################################################### Everything above may be tweaked manually before starting building and training;
        ################################################### Everything below will be automatically generated and shared across the class.

        # class instances of callbacks; used for monitoring training in real time/reviewing it afterwards..
        class _PrintEpochInfo(keras.callbacks.Callback):  # inherits from keras.callbacks.Callback,
            # which is a dummy class specifically designed for creating objects that goes into callbacks argument in tf.model.fit();
            # This is a local class that will not need to be reused outside of the function.
            start_time_global = self.timing["start_time_raw"]#grab the global start time from the timing dictionary above.
            def on_epoch_end(self, epoch, logs):  # redefine the function so that it prints only a dot, regardless of verbosity level.
                # ignore the logs (which logs the mae and mse)
                terminal_width = get_terminal_size().columns
                
                output_string = "{:>7} epochs finished;"\
                           "loss-value (mse) = {:0.9f};"\
                "validation loss-value (mse) = {:0.9f};"\
                "program has ran for = {:04.2f} s".format(
                        epoch + 1, logs["loss"], logs["val_loss"], time.time() - start_time_global)
                prompt_wider = "please make the terminal wider!"
                if terminal_width>=len(output_string):
                    print( output_string ,end="\r", flush=True)  # make sure the screen is wide enough to print all of this in a single line; otherwise it will overflow into the next line then the "\r" and flush operation will not extend back onto the first line, and the flush behaviour won't occur.
                elif len(prompt_wider)<=terminal_width<len(output_string):
                    print( prompt_wider, end="\r", flush=True)
                else:
                    pass #don't print anything.


        if not os.path.exists(".checkpoints/tb_logs/"): os.makedirs(".checkpoints/tb_logs/")
        self.callback_objects_available = {
            "PrintEpochInfo" : _PrintEpochInfo(), #just to print the epoch info to screen.
            "TensorBoard" : tf.keras.callbacks.TensorBoard(log_dir=".checkpoints/tb_logs/latest_run", histogram_freq=1), #overwrites the previously saved TensorBoard file.
            "EarlyStopping" : tf.keras.callbacks.EarlyStopping(patience=1000, restore_best_weights=True),
            "ProgbarLogger" : tf.keras.callbacks.ProgbarLogger(),
            "ReduceLROnPlateau": tf.keras.callbacks.ReduceLROnPlateau(),
        }
    
        self.keep_showing_figure = True #this has to be kept in order to make things simple and modular.
        
        self.folder = "test"

        #recording the model itself
        self.model = None

        # A list of variables used for sharing numerical data/object across methods.
        self.data_input = {
            # This dictionary only stores the corresponding data,
            # all of which are stored in the format of DataFrame
            "feature_before_preprocessing" : None,
            "train_feature" : None,
            "test_feature"  : None,

            "label_before_preprocessing"   : None,
            "train_label"   : None,
            "test_label"    : None,

            "true_spec" : None, # in usual operation, post-processing "test_label" will give "true_spec";
                                # i.e. the testing split of the trimmed "label_before_preprocessing" is identical to true_spec.
            "ref_spec"  : None, # a THIRD line to be plotted on the graph. This is only utilized when predicting the demo data.
            "ref_info"  : None, # dataframe from which the title text is loaded.

            "group_structure" : None,
            "response_matrix" : None,
        }

        self.evaluation_output = {
            # this is a hybrid dictionary that stores data in various formats (numpy.array, pandas.DataFrame, list).
            "hist_df" : None,
            "predicted_labels_array_before_post_processing": None, # Holds the prediction values (from file or from test set)
            "predicted_labels_array_after_post_processing" : None, 
            "error" : [],  # list of elementwise error
        }
        
    def interactive_neural_network_maker(self):
        key_input_prompt = "input the any key or attribute whose value that you'd like to change, or input 'c' to exit:"
        for d in self.settable_property_list:
            print("{0} :".format(d))
            print(getattr(self,d), "\n")
        while True:
            key_input = input(key_input_prompt)
            if key_input=="c":
                break
            for d in self.settable_property_list:
                val_input_prompt = "input the value for {0} as you would in python script ('quotes' around str, [brac] around lists, etc.):".format(d)
                if type(getattr(self,d))==dict:
                    keys = getattr(self,d).keys()
                    for k in keys:
                        if key_input.strip()==k:
                            val_input = convert_str_value(input(val_input_prompt))
                            dict_copy = getattr(self,d)
                            dict_copy[k]=val_input
                            setattr(self,d,dict_copy)
                            print(d, "now takes the value of ", dict_copy)
                elif key_input.strip()==d:
                    val_input = convert_str_value(input(val_input_prompt))
                    setattr(self,d,val_input)
                    print(d, "now takes the value of ", val_input)

    def try_to_update_attribute(self, test_k, value):
        if hasattr(self, test_k):
            setattr(self, test_k, value)
            return
        else:
            dictionaries = [ i for i in dir(self) if type( getattr(self,i) )==dict] #get the list of attributes which are dictianaries.
            for dic_name in dictionaries:
                if test_k in getattr(self,dic_name).keys(): # if the input key is found in the dictionary.
                    dic_copy = getattr(self, dic_name) #get a copy of the dictionary 
                    dic_copy[test_k] = value # change the corresponding value
                    setattr(self, dic_name, dic_copy)
                    return # only stop retun the method if we stop the case.
            raise KeyError("no attribute or key named", test_k)

    def load_data(self, csv_file, data_input_key):
        '''
        Retrieve data from .csv in the same directory without normalziation;
        Usual use case is
        nn.load_data("reaction_rate.csv","feature_before_preprocessing")
        nn.load_data("flux.csv", "labels_before_preprocessing")
        '''
        df = pd.read_csv(csv_file, delimiter=",", header=None, comment="#")

        # Error-checking:
        # Ensure that the data obtained are of the correct size before saving it as a class attribute.
        if "label" in data_input_key:
            opposite_key = data_input_key.replace("label", "feature")
        elif "feature" in data_input_key:
            opposite_key = data_input_key.replace("feature", "label")
        elif data_input_key=="ref_spec":
            opposite_key = "feature_before_preprocessing"
        elif data_input_key=="true_spec":
            opposite_key = "test_label"
        elif data_input_key=="group_structure":
            pass #ignore this case
        elif data_input_key=="response_matrix":
            df = pd.read_csv(csv_file, header=None, index_col=0) #redo the read, including the indices name for each row.
        elif data_input_key=="ref_info":
            df = pd.read_csv(csv_file, header="infer") #redo the read, including the column headers
            opposite_key="label_before_preprocessing"
        else:
            raise KeyError( "data_input_key='{0}' not found".format(data_input_key) )
        
        #by asserting that the opposite entry is of the same shape if it has been loaded:
        if data_input_key=="group_structure": #specific treatment for loading group_structure.
            num_boundaries = len(df.values.flatten())
            assert num_boundaries == max( np.shape(df) ), "The .csv where the group_structure is stored"\
                "must contain only a single line of data, stored vertically or horizontally"
            
            if type(self.data_input["label_before_preprocessing"])!=type(None):
                label_num_col = len(self.data_input["label_before_preprocessing"].columns)
            elif type(self.data_input["train_label"])!=type(None):
                label_num_col = len(self.data_input["train_label"].columns)
            try:
                assert num_boundaries== ( label_num_col+1 ), "there must be N+1"\
                    "group boundaries value provided for N flux values provided; "\
                    "But at the moment the group_structure has length = {1} "\
                    "which doesn't match the second dimension of train_label's boundary "\
                    "{0}".format( num_boundaries , np.shape(self.data_input["train_label"]) )
                    #Check that the shape of group_structure corresponds with the labels.
            except UnboundLocalError as E:
                if "label_num_col" in str(E):
                    pass # this means the group structure was loaded before "train_label" or "label_before_preprocessing" 
        elif data_input_key =="response_matrix":
            index_len, columns_len = df.shape
            if type(self.data_input["label_before_preprocessing"])!=type(None):
                label_col_len = len(self.data_input["label_before_preprocessing"].columns)
                assert label_col_len==columns_len, "number of columns in the response matrix({1}) must equal to the number of neutron groups({0})".format(label_col_len,columns_len)
            if type(self.data_input["feature_before_preprocessing"])!=type(None):
                feature_col_len = len(self.data_input["feature_before_preprocessing"].columns)
                assert feature_col_len==index_len, "number of activites in features({0}) must equal to the number of rows in the response matrix({1}).".format(feature_col_len, index_len)
        elif type(self.data_input[opposite_key]) != type(None):
            assert len(self.data_input[opposite_key].index) == len(df.index
            ),  "The entries in {0} must have one-to-one correspondance" \
                "with the entries in {1}. But they have shape {2} and {3}" \
                "respectively".format(data_input_key, opposite_key, np.shape(df),
                np.shape(self.data_input[opposite_key]))
        assert not (df.isnull().values.any()), "NaN value(s) found inside dataframe!"
        
        #saving the dataframe as an attribute to be used across the class.
        self.data_input.update({data_input_key:df})

    def _preprocess_numerical_values(self, df_or_array, datatype):
        assert (datatype=="feature") or (datatype=="label"), "The datatype must be specified either as 'label' or 'feature'."
        if datatype=="feature":
            if self.data_preparation_options["log_feature"]:
                df_or_array = np.log(df_or_array)
                df_or_array = np.clip(df_or_array, np.log( self.data_preparation_options["lower_limit"] ), None) #
        if datatype=="label":
            if not self.data_preparation_options["label_already_in_PUL"]:
                df_or_array = self._convert_to_PUL(df_or_array)
            if self.data_preparation_options["log_label"]:
                df_or_array = np.log(df_or_array)
                df_or_array = np.clip(df_or_array,np.log( self.data_preparation_options["lower_limit"] ), None)  #clip all values to above zero to prevent -inf's when taking log.
            if self.data_preparation_options["ft_label"]:
                df_or_array = fft(df_or_array)
        return df_or_array

    def trim_data(self): # self.data_reordering_options["cutoff"]
        '''
        Cut out unused data from self.data_input["feature"] and self.data_input["label"]
        using self.data_reordering_options["cutoff"]
        '''
        cutoff_point = self.data_reordering_options["cutoff"] #copying the global cutoff variable to a shorter expression.
        startoff_point = self.data_reordering_options["startoff"]
        if (cutoff_point==None) and (startoff_point==None): print("trim_data called but data is not trimmed since startoff and cutoff=None")
        self.data_input["feature_before_preprocessing"] = self.data_input["feature_before_preprocessing"][startoff_point:cutoff_point]
        self.data_input["label_before_preprocessing"] = self.data_input["label_before_preprocessing"][startoff_point:cutoff_point]
        if type(self.data_input["ref_spec"])!=type(None): #if ref_spec is not empty:
            self.data_input["ref_spec"] = self.data_input["ref_spec"][startoff_point:cutoff_point]
        if type(self.data_input["ref_info"])!=type(None):
            self.data_input["ref_info"] = self.data_input["ref_info"][startoff_point:cutoff_point]

    def shuffle(self): # self.data_reordering_options["shuffle_seed"]
        '''
        shuffle the *_before_preprocessing DataFrames in self.data_input to a random but reproducible order
        using self.data_reordering_options["shuffle_seed"]
        '''
        assert len(self.data_input["feature_before_preprocessing"])==len(
            self.data_input["label_before_preprocessing"]), "features and labelsmust have 1-to-1 correspondance."
        indices = np.arange(len(self.data_input["feature_before_preprocessing"]))
        if self.data_reordering_options["shuffle_seed"] != None:
            np.random.seed(self.data_reordering_options["shuffle_seed"])
            np.random.shuffle(indices)  # operate in-place
        else:
            print("shuffle is called but data is not shuffled since shuffle_seed=None")
        self.data_input["feature_before_preprocessing"]= self.data_input["feature_before_preprocessing"].loc[indices]
        self.data_input["label_before_preprocessing"]  = self.data_input["label_before_preprocessing"].loc[indices]
        if type(self.data_input["ref_spec"]) != type(None):
            self.data_input["ref_spec"] = self.data_input["ref_spec"].loc[indices]
        if type(self.data_input["ref_info"]) != type(None):
            self.data_input["ref_info"] = self.data_input["ref_info"].loc[indices]

    def split_into_sets(self): # self.data_reordering_options["train_split"]
        '''
        populate train_* and test_*
        by splitting *_before_preprocessing in two parts
        according to the fraction determined by self.data_reordering_options["train_split"]
        '''
        print("populating sets from *_before_preprocessing...")
        sample_size = len(self.data_input["feature_before_preprocessing"].index)
        # Use the first part as training data, the second part as
        num_train = round(self.data_reordering_options["train_split"] * sample_size)

        self.data_input["train_feature"] = self.data_input["feature_before_preprocessing"].iloc[:num_train]
        self.data_input["test_feature"] = self.data_input["feature_before_preprocessing"].iloc[num_train:]
        
        self.data_input["train_label"] = self.data_input["label_before_preprocessing"].iloc[:num_train]
        self.data_input["test_label"] = self.data_input["label_before_preprocessing"].iloc[num_train:]
        if type(self.data_input["ref_spec"])!=type(None):
            self.data_input["ref_spec"] = self.data_input["ref_spec"][num_train:]
        if type(self.data_input["ref_info"])!=type(None):
            self.data_input["ref_info"] = self.data_input["ref_info"][num_train:]

    def preprocess_input(self): # self.data_preparation_options
        '''
        Transform the numerical values inside the dataframe (in self.data_input) (reversibly)
        using the options listed in self.data_preparation_options
        '''
        #pick out ONLY the test_* and train_* labels and features; leaving the *_before_preprocessing alone.
        df_list = [ df_key for df_key in self.data_input.keys() if ("_feature" in df_key) or ("_label" in df_key) ]
        for k,v in self.data_input.items():
            if type(v) != type(None): # filter out all empty cases
                if "feature" in k:
                    v = self._preprocess_numerical_values( v , "feature")
                if "label" in k:
                    v = self._preprocess_numerical_values( v , "label")
                self.data_input.update({k:v})

    def _print_module_name(self): # dependent on whether build_model is called with print_pretty_logo=True or False.
        print("'|.   '|'                                  '||  ")
        print(" |'|   |    ....  ... ...  ... ..   ....    ||  ")
        print(" | '|. |  .|...||  ||  ||   ||' '' '' .||   ||  ")
        print(" |   |||  ||       ||  ||   ||     .|' ||   ||  ")
        print(".|.   '|   '|...'  '|..'|. .||.    '|..'|' .||. ")
        print("                                                ")
        print("                                                ")
        print("'|.   '|'           .                               '||      ")
        print(" |'|   |    ....  .||.  ... ... ...   ...   ... ..   ||  ..  ")
        print(" | '|. |  .|...||  ||    ||  ||  |  .|  '|.  ||' ''  || .'   ")
        print(" |   |||  ||       ||     ||| |||   ||   ||  ||      ||'|.   ")
        print(".|.   '|   '|...'  '|.'    |   |     '|..|' .||.    .||. ||. ")

    def build_model(self, print_pretty_logo=True): # self.hyperparameter
        '''
        using arguments saved in self.hyperparameter
        (which includes tf_seed ,hidden_layer ,act_func ,learning_rate ,loss_func ,metrics)
        a model is generated.
        '''
        tf.random.set_random_seed(self.hyperparameter["tf_seed"])

        # act_func should be a list
        act_func_iter = iter(
            [activations.linear, ] + self.hyperparameter["act_func"])  # ensure that the first layer is a purelin activation

        def get_next_activation_function():  # create a short method to iterate through activation functions
            try:
                act = next(act_func_iter)
            except StopIteration:
                act = activations.relu  # if user hasn't given enough activation functions, pad the rest using relu.
            return act

        neural_network_structure = []
        for n in self.hyperparameter["hidden_layer"]:
            if type(n) == int: # if it is an integer, interpret it as "numebr of nodes to insert into the next layer",
                neural_network_structure.append(layers.Dense(n, activation=get_next_activation_function())) # and match it to the next activation function on the list.
            elif type(n) == float:
                assert 0 < n < 1, "a float value is interpreted as a drop out rate, thus must be a fraction between 0 and 1."
                neural_network_structure.append(layers.Dropout(n))

        # The zeroth and last layer have linear activation functions
        # and shape corresponding to the input and output respectively.
        neural_network_structure.append(layers.Dense(len(self.data_input["train_label"].columns), activation=activations.linear))
        # first_layer_size = first integer value, otherwise if there are no hidden layers, then it equals the number of labels
        first_layer_size = len(self.data_input["train_label"].columns)
        for n in self.hyperparameter["hidden_layer"]:
            if type(n) == int:
                first_layer_size = n
                break
        # forcefully overwrite the first layer to have a purelin activation function,
        # and make sure the zeroth layer understands the input shape to be of shape=self.num_feature
        neural_network_structure[0] = layers.Dense(first_layer_size, input_shape=[len(self.data_input["train_feature"].columns)],
                                                   activation=activations.linear)
        #getting the loss function:
        loss_func = convert_str_to_loss_func(self.hyperparameter["loss_func"], self.data_input["response_matrix"], self.data_preparation_options["log_label"])
        
        model = keras.Sequential(neural_network_structure)
        model.compile(
            # loss="mean_squared_error",
            # loss="logcosh",
            loss=loss_func,
            # Mean squred error is the most sensible and widely chosen option among all loss functions in this case,
            # where where we're preforming a regression with no other boundary condition (e.g. area under graph =1) applied.
            # But perhaps later we may wish to define some functions to penalize for discontinuity between bins,
            # e.g.
            # def loss(x): return abs(np.diff(x)))
            optimizer=self.hyperparameter["optimizer"],  # use the RMS propagation algorithm listed above
            metrics=self.hyperparameter["metrics"] #******Look at changing the loss function and metrics!!!
            # save these parameters into the history object such that the accuracy of the NN to the validation set can be tracked.
        )
        if print_pretty_logo:
            self._print_module_name()
        # save these parameters as the class attributes
        self.optimizer = model.optimizer  # save the optimizer
        self.model = model

    def _print_params_as_dictionary(self): # dependent on whether train_model is called with print_dict_before_training=True or False.
        '''print all non-numerical parameters and hyperparameters to stdout'''
        dictionary_of_params = {}
        for k in self.settable_property_list:
            dictionary_of_params[k] = getattr(self, k)
        for k, v in dictionary_of_params.items():
            print(k, ":", v, "\n")

    def train_model(self, print_dict_before_training = True, verbose=0): # "num_epochs", "validation_split", callbacks_applied
        '''
        self.data_reordering_options["validation_split"]
        self.hyperparameter["num_epochs"]
        self.callbacks_applied, which contains the keys
            PrintEpochInfo
            TensorBoard
            EarlyStopping
            ProgbarLogger
            ReduceLROnPlateau
        usually only the first two are used.
        '''
        if print_dict_before_training:
            self._print_params_as_dictionary()

        print("using {0} training samples, which consist of a validation split = {1}, begin training for # epochs = {2}...".format(
            len(self.data_input["train_feature"].index), self.data_reordering_options["validation_split"], self.hyperparameter["num_epochs"]) ) 

        history = self.model.fit(

            self.data_input["train_feature"],
            self.data_input["train_label"]  ,
            
            epochs = self.hyperparameter["num_epochs"],
            validation_split = self.data_reordering_options["validation_split"],
            verbose = verbose,
            callbacks = [ self.callback_objects_available[k] for k in self.callbacks_applied ],

        )
        print("\ntraining complete!\n")  # skip a line to avoid overwriting the previous lines.

        hist_df = pd.DataFrame(history.history)
        epoch_of_interest = -1
        if 'EarlyStopping' in self.callbacks_applied:
            epoch_of_interest = hist_df["val_loss"].idxmin()
            self.hyperparameter["num_epochs"] = epoch_of_interest
        self.losses.update(dict(hist_df.iloc[epoch_of_interest]))
        
        hist_df['epoch'] = history.epoch # a column handle for plotting
        print(hist_df.tail())
        self.evaluation_output["hist_df"] = hist_df
        self.timing["run_time_seconds"] = time.time() - self.timing["start_time_raw"]

    def auto_generate_session_name(self): # add stuff in front of self.session_name
        all_non_dropout_layers = [l for l in self.hyperparameter["hidden_layer"] if type(l) == int]

        num_layer_str = str(len(all_non_dropout_layers)) + "_layer" # characterise the session by the number of layers used.
        
        datetime_str = time.strftime("%m%d_%H%M") + "_" # add the date and time to prevent name conflict

        #Sort these into folders according to their loss values.
        '''
        loss_value = list(self.evaluation_output["hist_df"]["val_loss"])[-1] #get the validation loss from the hist_df, which is guaranteed to have been generated and recorded at the training stage.
        if self.losses["loss"]!=0: #if the test loss has been recorded:
            loss_value = self.losses["loss"]
        '''
        self._evaluate_against_test_set() #force _evaluate_against_test_set to be run so that the self.losses['test_loss'] takes a non-zero (meaningful) value.
        rounddown_loss_magnitude = np.floor(np.log10(self.losses['test_loss'])).astype(int) #sort the .png's into folders according to their numbers.
        dir_str = "lossabove1e"+ str(rounddown_loss_magnitude) + "/"
        if not os.path.exists(dir_str): os.makedirs(dir_str)
        
        # sort by 1. loss value, 2. time,      3.hyperparameter,    4. custome name
        session_name = dir_str + datetime_str + num_layer_str + self.session_name
        
        self.session_name = session_name
        print("this session's details are saved in", session_name)

    def save_params_as_dictionary(self): #Overwrite old *_params.txt dictionary if present
        '''save all non-numerical parameters and hyperparameter into a .txt file.'''
        original_params_txt = self.session_name.split("layer")[-1]+"_params.txt" #always save at the CURRENT working directory; by ignoring all that *layer etc. stuff generated.
        f = open(original_params_txt, "w")
        f.write("{\n")
        def _write_datum(datum):
            if type(datum)==str:
                f.write("'")
                f.write(datum)
                f.write("'")
            else:
                f.write(str(datum))
        for k_1 in self.settable_property_list:
            entry = getattr(self, k_1)
            if type(entry)==dict:
                for k_2 in entry:
                    f.write(k_2)
                    f.write(" : ")
                    _write_datum(entry[k_2])
                    f.write(" ,\n")
            else:
                f.write(k_1)
                f.write(" : ")
                _write_datum(entry)
                f.write(" ,\n")
        f.write("}")
        f.close()

    def save_NN_weights(self):
        if not os.path.exists(".checkpoints/"): os.makedirs(".checkpoints/")  # make sure .checkpoint/ exist
        self.model.save_weights(".checkpoints/" + self.session_name.split("/")[-1] + ".h5")  # save the NN in the .checkpoints directory, ignoring the lines before it.

    def plot_history(self, show_plot_instead_of_saving = False):  # self.session_name+"_loss_value.png" will become the name of the saved plot
        num_metrics = len(self.hyperparameter["metrics"])+1 # loss + metrics = total number of metrics that will get outputted
        df = self.evaluation_output["hist_df"]  #get the hist_df in form of a shorter variable name.
        columns = df.columns[:-1] #ignoring the last column, which is the epoch number.
        optimal_epoch = self.hyperparameter["num_epochs"]
        
        fig, axes = plt.subplots(num_metrics, 1, sharex=True)  # Vertically stack the graphs
        if num_metrics==1:
            axes = [axes,] #wrap the single element into a list so that it can also be iterated through as well.

        axes[0].set_title("Performance of the neural network wrt. training progress")
        for i in range(num_metrics):
            train = columns[i]
            valid = columns[num_metrics+i]
            axes[i].set_ylabel( " ".join(columns[i].replace("squared","sq.").replace("absolute","abs.").replace("error","err.").split("_")) ) #replace the _ with space. and abbreviate.
            axes[i].semilogy(df["epoch"], df[ train ], label="train. error" )
            axes[i].semilogy(df["epoch"], df[ valid ], label="val. error")
            axes[i].legend()
            y_scatt = (df[train][optimal_epoch], df[valid][optimal_epoch])
            axes[i].scatter( np.ones(2)*optimal_epoch, y_scatt, color="r", marker="x")
        axes[-1].set_xlabel("# epochs")

        if show_plot_instead_of_saving:
            plt.show()
        else:
            plt.savefig(self.session_name + "_error_variation.png")
        plt.clf()
        plt.close()

    def _evaluate_against_test_set(self):
        # Print loss values when evaluated against test set
        losses_output = self.model.evaluate(self.data_input["test_feature"], self.data_input["test_label"]) #use tf.model.evalulate to get the loss values of the predictions.
        if type(losses_output) == list:
            for i in range(len(losses_output)):
                key = list(self.losses.keys())[i]
                self.losses.update({"test_"+key: losses_output[i]})
        else:
            self.losses.update({"test_loss": losses_output})
        self.save_params_as_dictionary() #overwrite existing dictionary with a very
        print("The loss values and other metrics when evaluated against the test set are obtained as {0}".format(self.losses) )
        # find the element-wise error
        self.evaluation_output["predicted_labels_array_before_post_processing"] = self.model.predict(self.data_input["test_feature"]) #use tf.model.predict to get the actual prediction themselves.
        self._postprocess_output() # popularte using the program self.postprocess_numerical...
        self.evaluation_output["error"] = self.evaluation_output["predicted_labels_array_before_post_processing"].flatten() - self.data_input["test_label"].values.flatten()
        #   use the difference between prediction and true values BEFORE postprocessing as the deviation/error list.
        #= self.evaluation_output["predicted_labels_array_after_post_processing"].flatten() - self.data_input["true_spec"].values.flatten()
        #   instead of using the difference of their respective values AFTER postprocessing.
 
    # compute how far off each label is, element-wise
    def plot_test_results_histogram(self, show_plot_instead_of_saving=False):
        prepend_in_bracket = ""
        if self.data_preparation_options["log_label"]:
            prepend_in_bracket += "log of "
        if self.data_preparation_options["ft_label"]:
            prepend_in_bracket += "fourier coefficients of "
        if len(self.evaluation_output["error"]) == 0:
            self._evaluate_against_test_set()#ensure that the error list isn't empty before continuing with the rest of the current method"
        plt.hist(self.evaluation_output["error"], bins=25)
        plt.suptitle("Prediction error on each element of the label, (i.e. " + prepend_in_bracket + "flux PUL"+ ")")
        plt.title("loss function(prediction, test_label)={0}".format(self.losses["test_loss"]))
        plt.xlabel("Error")
        plt.ylabel("Count")
        if show_plot_instead_of_saving:
            plt.show()
        else:
            plt.savefig(self.session_name + "_error_distribution.png")
        plt.clf()
        plt.close()

    def _postprocess_numerical_values(self, df_or_array, datatype): #datatype states whether it's 'label' or 'feature' that's being processed.
        assert (datatype=="feature") or (datatype=="label"), "The datatype must be specified either as 'label' or 'feature'."
        if datatype=="feature":
            if self.data_preparation_options["log_feature"]:
                df_or_array = e**df_or_array
        if datatype=="label":
            if self.data_preparation_options["ft_label"]:
                df_or_array = ifft(df_or_array)
            if self.data_preparation_options["log_label"]:
                df_or_array = e**df_or_array
            if not self.data_preparation_options["label_already_in_PUL"]:
                gs = self.data_input["group_structure"].values.flatten()#shorten the group structure list into 'gs'
                lethargy_span = np.diff(np.log(gs))                     #calculate the lethargy span of each bin
                df_or_array = df_or_array*lethargy_span                 #multiply the label (representing flux PUL) by lethargy span to get total flux instead.
        return df_or_array

    def _postprocess_output(self):
        self.evaluation_output["predicted_labels_array_after_post_processing"] = self._postprocess_numerical_values(self.evaluation_output["predicted_labels_array_before_post_processing"], "label")
        self.data_input["true_spec"] = self._postprocess_numerical_values(self.data_input["test_label"], "label") #un-log and un-fourier transform the data to get it back into the correct form.

    def _convert_to_PUL(self, flux):
        gs = self.data_input["group_structure"].values.flatten() #shorten the variable name into 'gs'
        lethargy_span = np.diff(np.log(gs)) #calculate the lethargy span of each bin
        flux = flux/lethargy_span
        return flux

    def _split_line_at_threshold(self, flux, upper_or_lower = "lower", threshold = 2):
        '''
        covnert flux to flux PUL,
        and chop it, leaving only the half that's above/below the threshold energy value.
        '''
        gs = self.data_input["group_structure"].values.flatten()
        # flux = self._convert_to_PUL(flux) #the flux has already been converted to PUL when inputting it.
        thres_ind = abs(gs - threshold).argmin() #find the index of the closest to the threshold
        if upper_or_lower =="lower":
            gs_cut = gs[:thres_ind+1]
            flux_cut = np.hstack([flux[0], flux[:thres_ind]])
        elif upper_or_lower=="upper":
            gs_cut = gs[thres_ind:]
            flux_cut = np.hstack([flux[thres_ind], flux[thres_ind:]])
        return gs_cut, flux_cut

    def _side_by_side_plot(self, press, ind, true_line, predicted_line , ref_spec_line=None, ref_info_line=None):
        '''
        make two plots,
            ax1 compares total flux in each bin according to bin number, by plotting predicted flux and true_flux side-by-side
            ax2 plots the flux in each bin.
        '''
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        # label the axes.
        ax1.set_xlabel("bin number"); ax1.set_ylabel("flux per unit lethargy per unit fluence (1/s)")
        ax2.set_xlabel("energy (MeV)")#; ax2.set_ylabel("flux (per unit lethargy)")
        # make the plot on the right a log-log plot,
        # ax1.set_yscale("log")
        ax2.set_yscale("log"); ax2.set_xscale("log")

        #add titles
        ax1.set_title("smooth plot of spectrum for comparison purpose."); ax2.set_title("log-log plot of spectrum")
        plt.suptitle("test spectrum " + str(ind))
        if type(ref_info_line)!=type(None):
            plt.suptitle(ref_info_line["title"]) #overwrite the suptitle
        # link up to the press() function (defined locally within the scope of self.compare_individual_spectra())
        fig.canvas.mpl_connect('key_press_event', press)
        #actual plotting
        ax2.step(*self._split_line_at_threshold(true_line, "upper", threshold=0), label="true fluence", alpha=0.8)
        ax2.step(*self._split_line_at_threshold(predicted_line, "upper", threshold=0), label="fluence predicted by NN", alpha=0.8)
        ax1.semilogy(true_line, label="true fluence", alpha=0.8)
        ax1.semilogy(predicted_line, label="fluence predicted by NN", alpha=0.8)
        #plotting the original spectrum if it exist.
        if type(ref_spec_line)!=type(None):
            ax2.step(*self._split_line_at_threshold(ref_spec_line), label="original flux before perturbation", alpha=0.8)
            ax1.semilogy(ref_spec_line, label="original flux before perturbation", alpha=0.8)
        #apply legends
        ax1.legend()
        ax2.legend()
        #maximize window
        mng = plt.get_current_fig_manager()
        if hasattr(mng, 'frame'):  # works with ubuntu
            mng.frame.Maximize(True)  # try to maximize the window
        else:
            try:
                mng.window.showMaximized()
                # mng.resize(*mng.window.maxsize())
            except:
                pass  # ignore this if python cannot maximize window; it has to be maximized manually.
        plt.show()
        plt.clf(); plt.close() #show an then close.
    
    def _C_E_plot(self, press, ind, true_line, predicted_line, ref_spec_line=None, ref_info_line=None, threshold=2):
        '''
        Plot the original spectrum and the NN's prediction on the same graph, and show the C/E value of each point below it.
        '''
        # naming axes according to the scale at x and y axes.
        fig, ([log_data, lin_data],
              [log_ce,     lin_ce]) = plt.subplots( 2, 2, sharex='col', sharey='row',
                                                    figsize=(12, 8),
                                                    gridspec_kw={   'width_ratios': [3, 2],
                                                                    'height_ratios': [6, 1]}, )
        plt.suptitle("test spectrum " + str(ind))
        if type(ref_info_line)!=type(None):
            plt.suptitle(ref_info_line["title"])
        log_data.set_xscale("log")
        lin_data.set_xscale("linear")
        log_data.set_yscale("log")
        log_data.set_ylabel("flux per unit lethargy per unit fluence (1/s)")
        log_ce.set_ylabel("calculated/expected (C/E)")
        unit="eV"
        if threshold<100: unit="MeV"
        log_ce.set_xlabel("E ({0}) on log scale".format(unit) )
        lin_ce.set_xlabel("E ({0})".format(unit) )
        log_ce.axhline(1,color="gray")
        lin_ce.axhline(1,color="gray")

        def plot_data(flux, label):
            log_data.step(*self._split_line_at_threshold(flux, "lower", threshold), label=label, alpha=0.8)
            lin_data.step(*self._split_line_at_threshold(flux, "upper", threshold), label=label, alpha=0.8)

        def plot_ce(ce):
            log_ce.scatter(*self._split_line_at_threshold(ce, "lower", threshold), marker="x", alpha=0.6) #fmt="C0x"
            lin_ce.scatter(*self._split_line_at_threshold(ce, "upper", threshold), marker="x", alpha=0.6) #fmt="C0x"

        plot_data(predicted_line, label="fluence predicted by NN")
        plot_data(true_line, label="true fluence")
        if type(ref_spec_line)!=type(None):
            plot_data(ref_spec_line, label="ref_spec_line_before_perturbation") #overwrite the suptitle
        plot_ce(predicted_line/true_line)

        # add legend to the graph
        log_data.legend()
        fig.tight_layout(rect=[0, 0, 1, 0.95]) # top right hand corner of 'rect'
        # has the coordinate (1,0.95) to prevent the suptitle clipping into the graph
        plt.savefig(self.session_name + "_test_" + str(ind).zfill(3) + "_fluence.png", dpi=180)
        plt.clf()
        plt.close()

    def _reaction_rate_compare(self, press, ind, true_line, predicted_line , ref_spec_line=None, ref_info_line=None, save_or_not=True):
        if type(ref_spec_line)!=type(None):
            ref_spec_line = np.array(ref_spec_line)
        
        response_matrix = np.array(self.data_input["response_matrix"])
        assert np.ndim(response_matrix)==2, "Please load the response matrix before doing self._reaction_rate_compare()!"
        true_activities = response_matrix.dot(true_line)
        predicted_activities = response_matrix.dot(predicted_line)
        num_activites = np.arange( len(response_matrix) )
        
        dist_in_log_space = np.log(predicted_activities/true_activities)
        mu = 0
        sigma = sum(np.sqrt( (dist_in_log_space-mu)**2 /len(dist_in_log_space) ))
        # chi2_dof = sum( (dist_in_log_space-0)**2 )/len(dist_in_log_space)
        # chi2txt = r"total $\frac{\chi^2}{DoF}$="+ str(chi2_dof) +"\n"+"assuming C/E is lognormally distributed around 1."
        if save_or_not:
            fig, (bar, ce) = plt.subplots(2,1, sharex=True,
                            gridspec_kw={'height_ratios': [6, 1]})

            reaction_names = [ i.replace("_",",") for i in self.data_input["response_matrix"].index ]

            ce.set_xticks(num_activites)
            ce.set_xticklabels(reaction_names, rotation=30, fontdict={"fontsize":8})

            num_bars = 2
            if type(ref_spec_line)!=type(None):
                ref_activites = response_matrix.dot(ref_spec_line)
                num_bars = 3
            width = 0.8/num_bars

            bar.set_ylabel("activity per unit fluence(1/s)")
            bar.bar(num_activites + width , predicted_activities, label="activities predicted by NN", width=-width, align="edge")
            bar.bar(num_activites, true_activities, label="true activities", width=-width, align="edge")
            if type(ref_spec_line)!=type(None):
                bar.bar(num_activites + 2*width, ref_activites, label= "original activities", width=-width, align="edge")
            bar.legend()
            bar.set_yscale("log")

            ce.axhline(1,color="gray")
            ce.scatter(num_activites, predicted_activities/true_activities, marker="x")
            ce.set_ylabel("C/E")
            
            sigmatxt = r"$\sigma$="+ str(sigma) +"\n"+"assuming C/E is normally distributed in log space, with a mean of 0."
            bar.set_title(sigmatxt)

            plt.suptitle( "test spectrum " + str(ind) )
            if type(ref_info_line)!=type(None):
                plt.suptitle(ref_info_line["title"]) #overwrite the suptitle
            
            # fig.text( 0.5, 0.0 , chi2txt, va="bottom", ha="center")
            # link up to the press() function (defined locally within the scope of self.compare_individual_spectra())
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(self.session_name + "_test_" + str(ind).zfill(3) + "_activities.png", dpi=100)
            plt.clf()
            plt.close()
        return sigma

    def _renormalize_prediction(self, fluxPUL):
        if self.hyperparameter["loss_func"]=="mean_pairwise_squared_error":
            n = np.ndim(fluxPUL)
            fluxPUL = ((fluxPUL).T/np.sum(fluxPUL, axis=n-1)).T
        return fluxPUL

    def compare_individual_spectra(self, using_simple_data=False, threshold = 2, save_C_E_plots = True, save_reaction_rate_comparisons=True, silent_mode=False):
        def press(event): #for stopping the plot comparison program when the key 'q' is pressed
            if event.key == 'q':
                self.keep_showing_figure = not self.keep_showing_figure
                print("Pressed 'q' to toggle self.keep_showing_figure to {0}".format(self.keep_showing_figure))

        does_ref_spec_exist = not (type(self.data_input["ref_spec"]) == type(None))

        # Need to compare the self.data_input["true_spec"] against the evaluation_output["predicted_labels_array_after_post_processing"].
        # Therefore the next part gets the evaluation_output["predicted_labels_array_after_post_processing"]
        if type(self.evaluation_output["predicted_labels_array_after_post_processing"])==type(None): #in case the _evaluate_against_test_set hasn't been ran
            #(such that _postprocess_output hasn't been called to populate evaluation_output properly)
            print("postprocessing test_label and predicted_labels to get true_spec and predicted spectrum respectively.")
            self._evaluate_against_test_set()

        #shorten the names
        true_spec = self.data_input["true_spec"].values
        predicted_labels = self.evaluation_output["predicted_labels_array_after_post_processing"]
        if does_ref_spec_exist:
            ref_spec = self.data_input["ref_spec"].values
        ref_info = self.data_input["ref_info"]
        does_ref_info_exist = not type(ref_info)==type(None)
        
        #covnert to PUL if not already in PUL.
        if (not using_simple_data) and (not self.data_preparation_options["label_already_in_PUL"]):
            predicted_labels = self._renormalize_prediction(self._convert_to_PUL(predicted_labels))
            true_spec = self._renormalize_prediction(self._convert_to_PUL(true_spec))#The true spectrum is left in the raw, non-PUL state until now.
            if does_ref_spec_exist:
                ref_spec = self._renormalize_prediction(self._convert_to_PUL(ref_spec))
        sigma_list = []
        for ind in range(len(predicted_labels)):
            if using_simple_data:
                fig, ax1 = plt.subplots()
                ax1.bar(np.arange(5), true_spec[ind], label="true fluence", width=0.4)
                ax1.bar(np.arange(5) + .4, predicted_labels[ind], label="fluence predicted by NN", width=0.4)
                ax1.legend()
                plt.suptitle("test spectrum " + str(ind))
                fig.canvas.mpl_connect('key_press_event', press)
                # link up to the press() function (defined locally within the scope of self.compare_individual_spectra())
                plt.show()
                plt.clf(); plt.close()
            else:
                ref_spec_line = None
                if does_ref_spec_exist:
                    ref_spec_line = pd.DataFrame(ref_spec).iloc[ind]

                ref_info_line = None
                if does_ref_info_exist:
                    ref_info_line = pd.DataFrame(ref_info).iloc[ind]
                if not silent_mode:
                    self._side_by_side_plot(press, ind, true_spec[ind], predicted_labels[ind], ref_spec_line=ref_spec_line, ref_info_line=ref_info_line)
                if save_C_E_plots:
                    self._C_E_plot(press, ind, true_spec[ind], predicted_labels[ind], ref_spec_line=ref_spec_line, ref_info_line=ref_info_line, threshold=threshold)
                sigma = self._reaction_rate_compare(press, ind, true_spec[ind], predicted_labels[ind], ref_spec_line=ref_spec_line, ref_info_line=ref_info_line, save_or_not = save_reaction_rate_comparisons)
                #will not save if save_reaction_rate_comparisons is False; in which case it will simply return the sigma to be appended to the sigma_list below:
                sigma_list.append(sigma)
            if not self.keep_showing_figure:
                break #condition to stop showing more figures if 'q' is pressed (self.keep_showing_figure is set by the locally defined function 'press')
        mean_sigma = np.mean(sigma)
        self.losses.update({'std_of_log_of_C_over_E_reaction_rates':mean_sigma})
        '''
        #THIS IS A BODGE to insert a line into the _params.txt.
        if not save_C_E_plots:
            original_params_txt = self.session_name.split("layer")[-1]+"_params.txt"
            with open(original_params_txt,"r") as f:
                lines = f.readlines()
            for i in range(len(lines)):
                if "}" in lines[i]:
                    brace_line_num = i
            with open(original_params_txt,"w") as f:
                [ f.write(l) for l in lines[:brace_line_num] ]
                f.write('std_of_log_of_C_over_E_reaction_rates : '+str(mean_sigma)+' ,\n')
                [ f.write(l) for l in lines[brace_line_num:] ]
        '''
        self.save_params_as_dictionary()

    def predict_from_additional_file(self, prediction_file_name):
        raw_unlabelled_features = pd.read_csv(prediction_file_name, header=None, comment="#")
        processed_unlabelled_features = self._preprocess_numerical_values(raw_unlabelled_features, "features")
        prediction_label_array_before_post_processing = self.model.predict(processed_unlabelled_features)
        return self._postprocess_numerical_values(prediction_label_array_before_post_processing, "feature")
    
    def plot_training_spectra(self,threshold):
        processed_train_label_df = pd.DataFrame(self.data_input["train_label"])
        max_num_plots=None
        if len(processed_train_label_df)>200: max_num_plots=50
        
        fig, (log_data, lin_data) = plt.subplots( 1, 2, sharey=True,
                                                    figsize=(12, 7),
                                                    gridspec_kw={'width_ratios': [3, 2]} )
        log_data.set_xscale("log")
        lin_data.set_xscale("linear")
        log_data.set_yscale("log")
        log_data.set_ylabel("flux per unit lethargy per unit fluence (1/s)")
        unit="eV"
        if threshold<100: unit="MeV"
        log_data.set_xlabel("E ({0}) on log scale".format(unit))
        lin_data.set_xlabel("E ({0})".format(unit))
        
        for flux in processed_train_label_df.iloc[:max_num_plots].iterrows():
            log_data.step(*self._split_line_at_threshold(flux[1], "lower",threshold=threshold), alpha=0.4)
            lin_data.step(*self._split_line_at_threshold(flux[1], "upper",threshold=threshold), alpha=0.4)
        plt.suptitle("Some of the spectra used to train the neural network with")
        plot_name = self.session_name+"_training_spectra"
        if hasattr(self, "train_label_file"): plot_name = ".".join(getattr(self, "train_label_file").split(".")[:-1])
        plt.savefig(plot_name+".png")