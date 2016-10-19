# DataWiz

Any data scientist would spend upwards of 50% of the time and effort in cleansing data. How great would it be if that were fully automated? This project aims to divert 20% of our efforts away from fancy ML models and into pure cleansing to producing an automated and repeatable process for data cleansing. Let's pause for a moment, on cutting the tree with an axe, and build a chainsaw.

Automated data cleansing for Decision Tree models (and similar models). Supports numpy and pandas.

Capabilities: Detecting Categorical columns and auto-encoding them, Detecting and removing headers from data, Detecting useless features, Handling missing values. More functionality is currently being developed.

Requires: 
Sklearn (plus an update to the preprocessing/label.py file) 
Pandas 
Numpy

INSTALL:
Copy the core/datawiz.py file into your project directory and import.

EXAMPLE:

import DataWiz

wiz = DataWiz(train_path='../.csv' ,
  test_path='../.csv',
  use='numpy', 
  target_col=-1,
  exclude_cols=[1,2,3],
  missing_values='fill',
  pds_chunksize=0)
 
X_clean, Y_clean = wiz.process()

wiz.read_test()

X_test_clean = wiz.process_test()
