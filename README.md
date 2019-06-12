#DataWiz

         ______       _____    _________    _____     ____       ___    ____   _________
        |   _  \     /     \  |__    ___|  /     \    |   |  _  |   |  |   |  |______   |
        |  | |  |   /  /_\  \    |  |     /  /_\  \   |   | / \ |   |  |   |       /  /
        |  | |  |  |   __   |    |  |    |   __   |   |   |/   \|   |  |   |      /  /
        |  |_|  |  |  |  |  |    |  |    |  |  |  |   |      /\     |  |   |     /  / ____   
        |______/   |__|  |__|    |__|    |__|  |__|   |_____/  \____|  |___|   |_________|


Any data scientist would spend upwards of 50% of the time and effort in cleansing data. How great would it be if that were fully automated? This project aims to divert 20% of our efforts away from fancy ML models and into pure cleansing to producing an ***automated and repeatable*** process for data cleansing. Let's pause for a moment, on cutting the tree with an axe, and build a chainsaw.

Automated data cleansing for Decision Tree models (and similar models). Supports numpy and pandas.

**Capabilities**: Detecting Categorical columns and auto-encoding them, Detecting and removing headers from data, Detecting & understanding date/time information, Detecting 'useless' features e.g. email addr or usernames, Handling missing values. More functionality is currently being developed.

**Requires**: 

 - sklearn (plus an update to the preprocessing/label.py file) 
 -  pandas 
 - numpy
 - datetime util

**Install**: 
You can install binaries from the Python Package Index via:

```pip install datawiz```

Conda installation not available at present.

**EXAMPLE**:
```python
import DataWiz
wiz = DataWiz(train_path='../.csv' , test_path='../.csv', use='numpy', target_col=-1, exclude_cols=[1,2,3],
              missing_values='fill', pds_chunksize=0)
X_clean, Y_clean = wiz.process() # This will remove headers, split the input and target columns,
                                 # determine useless features e.g. id or # email, and drop amy columns
                                 # specified in the "exclude_cols" argument of the class instantiation.
wiz.read_test()
X_test_clean = wiz.process_test()
```

**Arguments**:

**train_path**: path to the .csv file containing train data

**test_path**: path to the .csv file containing test data

**use**: 0 or 'numpy' for reading into a numpy array. 1 or 'pandas' for reading into a pandas dataframe. List type not currently supported for processing

**target_col**: the index of the target (or Y) column in the train data. Use -1 if it is the last coolumn

**exclude_cols**: an array in integers corresponding to the index of columns to be dropped from the train set. A similar drop will be applied to the test set.

**missing_values**: 'fill' to fill the missing values with the mode of the feature. 'drop' to drop the particular *rows* containing the missing value.  'drop' is not applied to the test set. It is applied as a 'fill' operation instead.

**pds_chunksize**: used when reading the .csv file with pandas. Recommended for very large datasets. See: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html     for details.

**dt_convert**: set to 1 to allow detection of columns containing date or time stamp values. set to 0 otherwise. default value is 1.

**advanced_ops**: this option tells the .process function to perform deeper (but potentially necessary)  data cleaning operations to prevent errors e.g. removing white space from text. Set to False to make code run faster, but True if you encounter errors.

**drop_cols**: Specifies whether recommended columns to be dropped are actually dropped autommatically. DataWiz detects columns with potentially irrelevant information for ML models e.g usernames, phone numbers etc. and recommends the user drop them.

**Data Structures**.

Where the class name is "wiz", the followig arrays are created:

**wiz.array**: The internal array (numpy/pandas) which is processed. Outputs from the .process() function are references to the cleaned version of this internal array

**wiz.array_test**: The internal array (numpy/pandas) which is processed for the test in a similar was the train set (above) was.

**wiz.dt_array**: This internal array is a list object which contains a number of numpy.ndarray or pandas.Series arrays (1-column vector of datetime or pandas.timeshamp types). i.e. dt_array = [ (numpy/pandas array of datetime variables 1), (numpy/pandas array of datetime variables 2) , (numpy/pandas array of datetime variables 3), (numpy/pandas array of datetime variables N)... ] where N is the number of colums in wiz.array that can be converted to a datetime object.

**wiz.dt_array_test**: Same as above, for the test set

