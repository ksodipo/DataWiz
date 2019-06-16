#DataWiz

         ______       _____    _________    _____     ____       ___    ____   _________
        |   _  \     /     \  |__    ___|  /     \    |   |  _  |   |  |   |  |______   |
        |  | |  |   /  /_\  \    |  |     /  /_\  \   |   | / \ |   |  |   |       /  /
        |  | |  |  |   __   |    |  |    |   __   |   |   |/   \|   |  |   |      /  /
        |  |_|  |  |  |  |  |    |  |    |  |  |  |   |      /\     |  |   |     /  / ____   
        |______/   |__|  |__|    |__|    |__|  |__|   |_____/  \____|  |___|   |_________|


Any data scientist would spend upwards of 50% of the time and effort in cleansing data. How great would it be if that were fully automated? This project aims to focus not on ML models but on every other aspect of data cleaning and processing, producing an ***automated and repeatable*** process for data cleansing. 

Automated data cleansing for Decision Tree models (and similar models). Built on top of the Pandas stack.

<table>
<tr>
  <td>Latest Release</td>
  <td>
    <a href="https://pypi.org/project/datawiz/">
    <img src="https://img.shields.io/badge/pip-v0.9-blue.svg" alt="latest release" />
    </a>
  </td>
</tr>
<tr>
  <td>Release Status</td>
  <td>
    <a>
    <img src="https://img.shields.io/badge/status-beta-brightgreen.svg" alt="status" />
    </a>
  </td>
</tr>
<tr>
  <td>License</td>
  <td>
    <a>
    <img src="https://img.shields.io/badge/license-MIT-orange.svg" alt="license" />
         </a>
</td>
</tr>

</table>

**Capabilities**: Detecting Categorical columns and auto-encoding them, Detecting and removing headers from data, Detecting & understanding date/time information, Detecting 'uninformaive or non-predictive' features e.g. email addr or usernames, Handling missing values. Automatically cleaning & transforming train and test data in a single function. More functionality is currently being developed.
<a>
<div align="center">
  <img src="https://github.com/ksodipo/DataWiz/blob/master/assets/img/datawiz_img.JPG"><br>
</div>
</a>

**Requires**: 

 - sklearn (plus an update to the preprocessing/label.py file) 
 - pandas 
 - numpy
 - datetime util

**Install**: 
You can install binaries from the Python Package Index via:

```pip install datawiz```

Conda installation not available at present.

**EXAMPLE**:
```python
import datawiz as dw

# 2 main functions exist
# 1st funtion, '.prescrbe' will load an excel file, classify and print columns into: numerical, categorical, datetime or #'uninformative.' 
'''It will also return the loaded data (train_data in e.g. below) and 4 list items: 
col_is_categorical (boolean list same length as the number of columns, True when column is categorical)
col_is_datetime (boolean list same length as the number of columns, True when column tells date and time) 
col_low_info (list of strings with the column names of uninformative columns. Recommended to drop these before using an ML algo)
col_good_info (list of strings with the column names of columns containing good info. Must be encoded before using an ML algo)
'''

train_data, col_categorical, col_datetime, col_low_info, col_good_info = dw.prescribe(train_path='../.csv' , test_path='../.csv', pds_chunksize=0)
              
X_clean,Y_clean,X_test,[encoders, encoded_cols],[dt_arrays, dt_cols] = 
dw.process(train_path=None,test_path=None,target_col=-99,exclude_cols=[],missing_values='fill',pds_chunksize=0,data_has_ws = True,encode_categories=True,dt_convert=True,drop_low_info_cols=True) 
                                 # This will remove headers, split the input and target columns,
                                 # remove useless features e.g. id or # email, and drop any columns
                                 # specified in the "exclude_cols" argument of the class instantiation.
                                 # Returns the encoders and encoded columns (both list objects) which specify the LabelEncode object and                                  # the name of the encoded column respectively. This should be used to encode columns in the test data

```

**Quick Use Example**

Perform all data cleaning and modelling in 4 lines:
```python
import datawiz as dw
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

X_clean, Y_clean, X_test, e, d = dw.process(train_path='C:/Users/.../Downloads/data_folder/train.csv',
                                test_path='C:/Users/.../Downloads/data_folder/test.csv',
                                exclude_cols=[0],target_col=1,
                                dt_convert=True)

model = DecisionTreeClassifier()

model.fit(X,Y)

predictions = model.predict(X_test)
```
**Return Values**

 **dw.prescribe**: (see above)
 
 **dw.process**: 
 
 X_clean = Pandas.DataFrame object containing input/training data (with target column removed) and specified data preparation operations applied
 
 Y_clean = target variable in input/training data i.e. variable to be predicted, in Pandas format
 
 X_test = test data with specified data cleaning operations applied
 
 encoders = A list of LabelEncode objects. 
 
 encoded_columns = A list of column names e.g. ['col1','col x','col y'] such that the column names correspond to the LabelEncode objects in the encoders array above
 
 dt_arrays = A list of pandas Datetime series
 
 dt_cols = A list of column names such that each columnn name corresponds to the Datetime object in the dt_arrays list
 
**Arguments**:

**train_path**: path to the .csv file containing train data

**test_path**: path to the .csv file containing test data

**target_col**: the column index of the target (or Y) column in the train data. Use -1 if it is the last coolumn

**exclude_cols**: an array in integers corresponding to the index of columns to be dropped from the train set. A similar drop will be applied to the test set. 

**missing_values**: 'fill' to fill the missing values with the mode of the feature. 'drop' to drop the particular *rows* containing the missing value.  'drop' is not applied to the test set. It is applied as a 'fill' operation instead.

**pds_chunksize**: used when reading the .csv file with pandas. Recommended for very large datasets. See: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html     for details.

**encode_categories**: set to True to allow automatic encoding of categorical columns into numbers e.g. 'Spring', 'Summer', 'Autumn' -->> 1,2,3

**dt_convert**: set to True to allow detection of columns containing date or time stamp values. If True, a list of [dt_array, dt_cols] is returned, where dt_array is a list containing the date_time columns (in pandas.Series) from the data. dt_cols specifies the corresponding colum name 

**drop_low_info_cols**: Set to True/False. Specifies whether recommended columns to be dropped are actually dropped autommatically. DataWiz detects columns with potentially irrelevant information for ML models e.g usernames, phone numbers etc. and recommends the user drop them.

