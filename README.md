**DataWiz**

Any data scientist would spend upwards of 50% of the time and effort in cleansing data. How great would it be if that were fully automated? This project aims to divert 20% of our efforts away from fancy ML models and into pure cleansing to producing an ***automated and repeatable*** process for data cleansing. Let's pause for a moment, on cutting the tree with an axe, and build a chainsaw.

Automated data cleansing for Decision Tree models (and similar models). Supports numpy and pandas.

**Capabilities**: Detecting Categorical columns and auto-encoding them, Detecting and removing headers from data, Detecting useless features, Handling missing values. More functionality is currently being developed.

**Requires**: 

 - sklearn (plus an update to the preprocessing/label.py file) 
 -  pandas 
 - numpy

**Install**: Copy the core/datawiz.py file into your project directory and import. Also consider replacing the sklearn.../preprocessing/label.py file to deal with cases where a new categorical variable is found in the test set.

**EXAMPLE**:

    import DataWiz
    
    wiz = DataWiz(train_path='../.csv' , test_path='../.csv', use='numpy', target_col=-1, exclude_cols=[1,2,3], missing_values='fill', pds_chunksize=0)
    
    X_clean, Y_clean = wiz.process() #This will remove headers, split the input and target columns, determine useless features e.g. id or # email, and drop amy columns specified in the "exclude_cols" argument of the class instantiation.
    
    wiz.read_test()
    
    X_test_clean = wiz.process_test()

**Arguments**:

**train_path**: path to the .csv file containing train data

**test_path**: path to the .csv file containing test data

**use**: 0 or 'numpy' for reading into a numpy array. 1 or 'pandas' for reading into a pandas dataframe. List type not currently supported for processing

**target_col**: the index of the target (or Y) column in the train data. Use -1 if it is the last coolumn

**exclude_cols**: an array in integers corresponding to the index of columns to be dropped from the train set. A similar drop will be applied to the test set.

**missing_values**: 'fill' to fill the missing values with the mode of the feature. 'drop' to drop the particular *rows* containing the missing value.  'drop' is not applied to the test set. It is applied as a 'fill' operation instead.

**pds_chunksize**: used when reading the .csv file with pandas. Recommended for very large datasets. See: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html     for details.

**advanced_ops**: this option tells the .process function to perform deeper (but potentially necessary)  data cleaning operations to prevent errors e.g. removing white space from text. Set to False to make code run faster, but True if you encounter errors.
