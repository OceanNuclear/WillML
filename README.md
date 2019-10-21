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

1. (optional) Shuffle ```RawData.csv``` using ```python Shuffle.py <directory name>``` + ```<optional parameters>``` (see below).
	```test_data.csv``` and ```train_data.csv``` will be automatically generated in that directory as a result.
	Note that one can add, directly, the following optional parameters behind the command above:
	- ```TRAIN_FRAC =``` decimal number between 0 to 1 (default = 0.5): e.g. ```python Shuffle.py 1/ TRAIN_FRAC = 0.3```
	- ```TRAIN_THRES =``` integer ≥ 0 (default = 1): e.g. ```python Shuffle.py 1/ TRAIN_THRES = 3```

The program is written in such a way that ensures all unlabelled data falls within the testing set; and among all labelled samples, for each category, a number of samples = ```TRAIN_THRES``` is present in the training set before any more items of such category starts to appear in the testing set.

2. Turn them into numerical representations by ```python Embed.py <directory name>``` with the option:
	- ```EMBED_BY_COUNTING = <boolean>``` (```default = True```)

3. Carry out the learning and prediction stage using SVM's:
	```python MachineLearning.py <directory name>```
	- ```LOGSVM = <boolean>``` (```default = False``` to save time). If set to True, it will also calculate the prediction results using a logistic regression support vector machine as well as the default linear SVM. The latter is much slower to compute.

4. Return them back into human readable format using ```python HumanReadable.py <directory name>```
    - ```MERGE_UNSCRAMBLE = <boolean>``` (```default = True```). If True, will try to find ```order_original.pkl``` (created in step 1), and unscramble the dataframe back into the order that it appeared in ```RawData.csv```.

After all these steps, a lot of \*.pkl files will be saved in the directory. Don't worry, you can delete them all. They are merely python variables, stored as ```.pkl``` objects.

## Explanation
Will explain the maths in a bit. Just know that it works right now.
