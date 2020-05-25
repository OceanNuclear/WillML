# Item Categorization Using Machine Learning
Input: a labelled training set, and a partially/entirely unabelled testing set.
Output: the two most likely categories that the items will fall into, along with the likelihood values.

## Spreadsheet (CSV) format
Column 0 (A) is the item index. Column 1 and 2 (B-C) are the item description. Column D is the category, which must be partially filled for the ```RawData.csv``` or totally filled for ```train_data.csv``` (see step 1 and 2 of [the section below](#Usage))

DO NOT put a header (title) row.

## Usage
Create a directory: e.g. ```1/```
Store, in that directory, either
- a ```RawData.csv```,
- or a pair of csv: ```test_data.csv``` and ```train_data.csv``` (skip straight to step 2. below.)

1. (optional) Shuffle ```RawData.csv``` using
	-```python Shuffle.py <directory name>``` + ```<optional parameters>``` (see below).
	```test_data.csv``` and ```train_data.csv``` will be automatically generated in that directory after running ```Shuffle.py```.

	Note that one can add, directly, the following optional parameters behind the command above:
	- ```TRAIN_FRAC =``` decimal number between 0 to 1 (default = 0.5): e.g. ```python Shuffle.py 1/ TRAIN_FRAC = 0.3```
	- ```TRAIN_THRES =``` integer ≥ 0 (default = 1): e.g. ```python Shuffle.py 1/ TRAIN_THRES = 3```

The program is written in such a way that ensures all unlabelled data falls within the testing set; and among all labelled samples, for each category, a number of samples = ```TRAIN_THRES``` is present in the training set before any more items of such category starts to appear in the testing set.

2. Turn them into numerical representations () by
	- ```python Embed.py <directory name>```

	The following argument can be changed at the source-code ```Embed.py```:
	- ```EMBED_BY_COUNTING = <boolean>``` (```default = True```)

3. Carry out the learning and prediction stage using SVM's:
	- ```python MachineLearning.py <directory name>```

	The following argument can be changed at the source-code ```MachineLearning.py```
	- ```LOGSVM = <boolean>``` (```default = False``` to save time). If set to True, it will also calculate the prediction results using a logistic regression support vector machine as well as the default linear SVM. The latter is much slower to compute.

4. Convert them back into human readable format using
	-```python HumanReadable.py <directory name>```

	The following argument can be changed at the source-code ```HumanReadable.py```
	- ```MERGE_UNSCRAMBLE = <boolean>``` (```default = True```). If True, will try to find ```order_original.pkl``` (created in step 1), and unscramble the dataframe back into the order that it appeared in ```RawData.csv```.

After all these steps, a lot of \*.pkl files will be saved in the directory. Don't worry, you can delete them all. They are merely python variables, stored as ```.pkl``` objects.

## Explanation
This program "learns" to associate words with categories.
E.g. The entry ```Element Jaywalker backpack in black``` will be broken down into 5 words, ```Element``` ```Jaywalker``` ```backpack``` ```in``` ```black```; and this entry has the category label ```backpack```. The will associate the category ```backpack``` to these 5 words.

This process is repeated once for each of the ```train_data.csv``` entry, strengthening/deminishing the association between occurance of words and the category labels.

E.g. if word ```bracelet``` only appeared once in the entire ```train_data.csv``` and that entry is given the category label ```Jewellery```, then when a new unlabelled entry with only the word ```bracelet``` in its name appears, the program will with give this entry the category label ```Jewellry``` with a very high confidence score.
