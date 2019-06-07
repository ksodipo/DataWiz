# Authors: Koye Sodipo <koye.sodipo@gmail.com>
# License: BSD 3 clause

import numpy
import pandas
import csv
import gc
from sklearn import preprocessing
from random import randint
from scipy import stats
from dateutil.parser import parse


# Two fundamental problems with determining whether 1st row is header:
# 1.) If all  elements (including the header) are numbers, code will think header is just any other data point
# 2.) If all elements (including the header) are strings, code will think
# header is just any othe data point. This can be solved if we assume the
# header title is unique to the entire column (i.e. occurs only once as
# header).


class DataWiz:
    def __init__(
            self,
            train_path=None,
            test_path=None,
            use=0,
            target_col=-99,
            exclude_cols=[],
            missing_values='fill',
            dt_convert=1,
            pds_chunksize=0,
            advanced_ops=True,
            drop_cols=False):

        # Default settings
        self.file_path = train_path
        self.test_file_path = test_path
        self.to_use = use
        self.use_numpy = True
        self.use_pandas = False
        self.use_list = False
        self.target_column = target_col
        self.exclude_columns = exclude_cols
        self.test_split = 0.2
        self.missing_vals = missing_values
        self.pd_chunksize = pds_chunksize
        self.dt_convert = dt_convert

        # Removes white space in string columns, datetime conversion
        self.advanced_ops = advanced_ops
        # Specifies whether recommended columns to be dropped are actually
        # dropped autommatically
        self.drop_cols = drop_cols

        # Advanced Defult settings (not editable through arguments)

        # should the date parser consider the first number group ('09') in '09/12/2010' as the day?
        self.dayfirst = True

        self.array = []
        self.array_test = []
        ans = -1
        self.accum = []
        self.header_or_not = []
        self.col_is_categorical = []
        self.col_is_datetime = []
        self.encoders = []
        self.header = []

        self.dt_array = []
        self.dt_array_test = []

        while True:
            try:
                if self.file_path is None:
                    self.file_path = input(
                        'Enter train file path (surround with quotes)   :')
                if self.to_use is None:
                    self.to_use = input(
                        'Enter 0 for numpy, 1 for pandas and 2 for list:  ')
                if self.target_column == -99:
                    self.target_column = input(
                        'Enter index of the the target column. Enter "None" if no target:  ')
                if self.exclude_columns == []:
                    while True:
                        excl = input(
                            'List the index of columns to exclude. Enter "None" to quit...')
                        if excl is None:
                            break
                        self.exclude_columns.append(excl)

                ans = 1
            except:
                raise NameError('Please enter valid answers')

            if ans == 1 or ans == 0:
                break

        self.use_numpy = True if (
            self.to_use == 0 or self.to_use == 'numpy') else False
        self.use_pandas = True if (
            self.to_use == 1 or self.to_use == 'pandas') else False
        self.use_list = True if (
            self.to_use == 2 or self.to_use == 'list') else False

        if self.use_numpy:
            csv_iter = csv.reader(open(self.file_path, 'rb'))
            data = [row for row in csv_iter]
            self.array = numpy.array(data)
            del data
            gc.collect()

        elif self.use_pandas:
            if self.pd_chunksize > 0:
                self.array = None
                for i, chunk in enumerate(pandas.read_csv(
                        self.file_path, chunksize=self.pd_chunksize, low_memory=False)):
                    if self.array is None:
                        self.array = chunk.copy()  # not simply a reference to it
                    else:
                        self.array = pandas.concat([self.array, chunk])
                    del chunk
                    gc.collect()

            else:
                self.array = pandas.read_csv(self.file_path)

        elif self.use_list:
            csv_iter = csv.reader(open(self.file_path, 'rb'))
            self.array = [row for row in csv_iter]

    def process(self):

        def is_datetime(arr):
            total = len(arr)
            accum = []
            for item in arr:
                item = str(item)
                if len(item) >= 6:  # parse() mistakes strings like '13', '3' etc for dates
                    # print(item)
                    try:
                        parse(item)
                        accum.append(1)
                    except ValueError:
                        accum.append(0)

            if sum(accum) == total:
                return True
            else:
                return False

        if isinstance(self.array, numpy.ndarray):

            try:
                rng = xrange(0, len(self.array[0, 0:]))
            except NameError:
                rng = range(0, len(self.array[0, 0:]))

            for column in rng:  # Test each column
                # initialize an array of 40 valid indexes to randomly sample in
                # a given column. We reset the first value of the array to 0 to
                # test the potential header row

                try:
                    test_value_types = [randint(1, len(self.array) - 1) for i in xrange(0, 41)]
                except NameError:
                    test_value_types = [randint(1, len(self.array) - 1) for i in range(0, 41)]

                test_value_types[0] = 0
                self.accum = []  # assumes labels are not integers
                for index in test_value_types:
                    try:
                        # Better to use float() than int() as int('123.45')
                        # will throw a ValueError, giving the impression that
                        # we're dealing with a string
                        float(self.array[0:, column][index])
                        # if self.array[0][column]=='Fare':
                        # print('Hit',index,self.array[0:,column][index])
                        self.accum.append(1)
                    except:
                        # if self.array[0][column]=='Fare':
                        #       print 'Miss',index,self.array[0:,column][index]
                        self.accum.append(0)
                        raise ValueError

                if isinstance(self.array[0, column], numpy.string_) and sum(self.accum) < 41 and sum(self.accum) > 0:
                    # This logic fails though, if the entire dataset is made of
                    # categorical strings and has NO headers. It will still
                    # assume 1st item is header regardless.
                    self.header_or_not.append(True)
                else:
                    self.header_or_not.append(False)

                # if the sum of 1s (instances where we found a number) is less
                # than 35, it's probably a categorical column
                if sum(self.accum) < 35:
                    # we might have dirty e.g. nan or a one off string  or
                    # missing data which would trick the code into thinking the
                    # column is categorical.
                    self.col_is_categorical.append(True)
                else:
                    self.col_is_categorical.append(False)

                # Decipher whether this column is datetime. Necessary to remove
                # '0' index
                test_value_types.pop(0)
                self.col_is_datetime.append(is_datetime(
                    self.array[test_value_types, column]))

            # Here we decide whether or not the data has headers
            is_header = True if True in self.header_or_not else False

            if is_header:
                # convert the numpy columns that were incorrectly assumed to be strings (and are numbers) to numbers...
                #  Actually, this isn't necessary as the sklearn DT converts all strings to floats
                # if header, split header from data. Then detect categorical
                # columns. create label encoder for that
                self.header = self.array[0]
                print('Header Row: ', self.header)
                ndata = self.array[1:]
            else:
                ndata = self.array[0:]
                # Make the header array out of indexes
                try:
                    self.header = [str(i) for i in xrange(0, len(self.array[0]))]
                except NameError:
                    self.header = [str(i) for i in range(0, len(self.array[0]))]

            # Handle missing values
            if self.missing_vals == 'fill':

                try:
                    rng = xrange(0, len(self.array[0, 0:]))
                except NameError:
                    rng = range(0, len(self.array[0, 0:]))

                for column in rng:
                    if self.col_is_categorical[column]:
                        mode = stats.mode(ndata[column])[0][0]
                        # ndata[column] = ndata[column].fillna(mode)
                    else:
                        try:
                            mean = numpy.mean(ndata[column])
                            # ndata[column] = ndata[column].fillna(mean)
                        except:
                            raise TypeError

            elif self.missing_vals == 'drop':
                n = 0
                # ndata = ndata.dropna('rows')

            try:
                rng = xrange(0, len(self.array[0, 0:]))
            except NameError:
                rng = range(0, len(self.array[0, 0:]))

            for column in rng:
                # if it's categorical but not a date
                if self.col_is_categorical[column] and self.col_is_datetime[column] is False:
                    # convert to number labels using LabelEncode
                    encoder = preprocessing.LabelEncoder()
                    if self.advanced_ops:  # remove leading or trailing spaces
                        ndata[:, column] = numpy.char.strip(ndata[:, column])
                    encoder.fit(ndata[:, column])
                    no_of_unique = len(encoder.classes_)
                    # if we have so many unique labels relative to the number
                    # of rows, it's probably a useless feature or an identifier
                    # (both usually) e.g. a name, a product description, ticket number, phone number,
                    # staff ID. More feature engineering usually required.
                    # Unsuprvised PCA perhaps.
                    if float(no_of_unique) / float(len(self.array)) > 0.25:
                        # ... also, even if we accidentally rule out a legitimate feature, the metric being > 0.25
                        #  would probably be a feature that'll cause overfitting
                        encoder = 'Column propably not useful'
                        self.encoders.append(encoder)
                        # , float(no_of_unique),float(len(self.array))
                        print('Consider dropping the column: ', self.header[column])
                        if self.drop_cols:
                            self.exclude_columns.append(column)
                    else:
                        # this back references and actually modifies array
                        ndata[:, column] = encoder.transform(ndata[:, column])
                        # output of encoder.transform is a numpy.ndarray, FYI
                        self.encoders.append(encoder)

                else:
                    self.encoders.append('Not a Category')

                # Attach a datetime object for each column. Has to be an
                # external array as numpy arrays can't hold datetime objects
                if self.dt_convert == 1:
                    if self.col_is_datetime[column]:
                        # Or make it a numpy array: numpy.array([parse(i) for i
                        # in ndata[:,column]])
                        self.dt_array.append(numpy.array(
                            [parse(i, dayfirst=self.dayfirst) for i in ndata[:, column]]))
                        # Makes a list of numpy arrays containing datetime
                        # objects.
            if self.target_column != 99 or self.target_column is not None:
                Y = ndata[:, self.target_column]

            if self.target_column == -1:
                # The extractor wouldn't recognize -1 two lines from here.
                self.target_column = len(self.array[0, 0:]) - 1
            # get a list of all valid indexes of columns

            try:
                array_of_col_index = [n for n in xrange(0, len(self.array[0, 0:]))]
            except NameError:
                array_of_col_index = [n for n in range(0, len(self.array[0, 0:]))]

            # this way, we only extract the train columns
            X = ndata[:, [i for i in array_of_col_index if (
                i != self.target_column and i not in self.exclude_columns)]]

            ##########################################################################
        if isinstance(self.array, pandas.core.frame.DataFrame):

            try:
                rng = xrange(0, len(self.array.columns))
            except NameError:
                rng = range(0, len(self.array.columns))

            for column in rng:  # Test each column
                # initialize an array of 40 valid indexes to randomly sample in
                # a given column. We reset the first value of the array to 0 to
                # test the potential header row

                try:
                    test_value_types = [randint(1, len(self.array) - 1) for i in xrange(0, 41)]
                except NameError:
                    test_value_types = [randint(1, len(self.array) - 1) for i in range(0, 41)]

                test_value_types[0] = 0
                self.accum = []  # assumes labels are not integers
                for index in test_value_types:
                    try:
                        float(self.array.loc[index][column])
                        self.accum.append(1)
                    except:
                        self.accum.append(0)
                        raise ValueError

                # if first item in row is a string and the rest are numbers
                # (i.e. sum of accum falls short of 40), assume that's a
                # header.
                if isinstance(self.array.loc[0][column],str) and sum(self.accum) < 41 and sum(self.accum) > 0:
                    # This logic fails though, if the entire dataset is made of
                    # categorical strings and has NO headers. It will still
                    # assume 1st item is header regardless.
                    self.header_or_not.append(True)
                else:
                    self.header_or_not.append(False)

                # if the sum of 1s (instances where we found a number) is less
                # than 35, it's probably a categorical column
                if sum(self.accum) < 35:
                    self.col_is_categorical.append(True)

                else:
                    self.col_is_categorical.append(False)

                test_value_types.pop(0)
                col_name = self.array.columns[column]
                # if .loc[x][y], where x is not a single int index, y MUST be
                # the name of the column, not simply an index
                self.col_is_datetime.append(is_datetime(
                    self.array.loc[test_value_types][col_name]))

            # Here we decide whether or not the data has headers
            is_header = True if True in self.header_or_not else False

            if is_header:
                # convert the pandas columns that were incorrectly assumed to be strings (and are numbers) to numbers...
                #  Actually, this isn't necessary as the sklearn DT converts all strings to floats
                # if header, split header from data. Then detect categorical
                # columns. create label encoder for that
                self.header = self.array[0:1]
                # print('Header Row: ',self.header)
                ndata = self.array[1:]
            else:
                ndata = self.array[0:]

            # Handle missing values
            if self.missing_vals == 'fill':
                for index, column in enumerate(self.array.columns):
                    if self.col_is_categorical[index]:
                        mode = stats.mode(ndata.loc[:][column])[0][0]
                        ndata[column] = ndata[column].fillna(mode)
                    else:
                        print(column)
                        try:
                            mean = numpy.mean(
                                ndata[column][
                                    pandas.notnull(
                                        ndata[column])])
                            ndata[column] = ndata[column].fillna(mean)
                        except:
                            raise TypeError

            elif self.missing_vals == 'drop':
                ndata = ndata.dropna('rows')

            for index, column in enumerate(self.array.columns):
                if self.col_is_categorical[index] and self.col_is_datetime[index] is False:
                    # convert to number labels using LabelEncode
                    encoder = preprocessing.LabelEncoder()
                    if self.advanced_ops:
                        ndata[column] = ndata[column].str.strip()
                    encoder.fit(ndata[column])
                    no_of_unique = len(encoder.classes_)
                    # if we have so many unique labels relative to the number
                    # of rows, it's probably a useless feature or an identifier
                    # (both usually) e.g. a name, ticket number, phone number,
                    # staff ID. More feature engineering usually required.
                    # Unsuprvised PCA perhaps.
                    if float(no_of_unique) / \
                            float(len(self.array[column])) > 0.25:
                        # ... also, even if we accidentally rule out a legitimate feature, the metric being > 0.25
                        #  would probably be a feature that'll cause overfitting
                        encoder = 'Category Not Relevant'
                        print('Consider dropping the column ', self.array.columns[index])
                        if self.drop_cols:
                            self.exclude_columns.append(index)
                    else:
                        # this back references and actually modifies array
                        ndata.loc[:][column] = encoder.transform(ndata[column])
                    # output of encoder.transform is a numpy.ndarray, FYI
                    self.encoders.append(encoder)
                    # In test test, be sure to only transform where
                    # col_is_categorical AND encoder != None i.e. 1st instance
                    # of True in col_is_categorical checks ast index of
                    # encoders array. 2nd checkeck 2nd etc..
                else:
                    self.encoders.append('Not a Category')

                # Attach a datetime object for each column.
                if self.dt_convert == 1:
                    if self.col_is_datetime[index]:
                        # creates a list of pandas series containing class
                        # 'pandas.tslib.Timestamp' objects
                        self.dt_array.append(pandas.Series(
                            [parse(i, dayfirst=self.dayfirst) for i in ndata[column]]))

            # Get the pandas names of columns before removing target col. 1. to preserve index. 2. Pandas doesn't like
            # dealing with indexes. Prefers names
            col_names_excl = []

            if self.exclude_columns is not None:
                for ind in self.exclude_columns:
                    col_names_excl.append(ndata.columns[ind])

            if self.target_column == -1:
                # .pop sometimes can't deal with -1 as an index
                self.target_column = len(ndata.columns) - 1
            Y = []
            if self.target_column != -99 and self.target_column is not None:
                Y = ndata.pop(self.array.columns[self.target_column])

            # disposing of columns not needed, considering memory
            for i in col_names_excl:
                garbage = ndata.pop(i)
                del garbage

            gc.collect()
            X = ndata

        self.is_processed = True
        return X, Y  # This is great because X is only a reference to the array object created outside of the function
        # Our previous setting of ndata to an index of array persists as a
        # global rule. If global array modified out of func, ndata, X changes
        # too.

    def read_test(self):
        while True:
            ans_t = 1
            try:
                if self.test_file_path is None:
                    self.test_file_path = input(
                        'Enter file path (surround with quotes)   :')
                break
            except:
                raise NameError

        # User can reset these by manipulating the class objects directly
        # before calling read_test()
        if self.use_numpy:
            csv_iter = csv.reader(open(self.test_file_path, 'rb'))
            data = [row for row in csv_iter]
            self.array_test = numpy.array(data)
            del data
            gc.collect()

        elif self.use_pandas:
            if self.pd_chunksize > 0:
                self.array_test = None
                for i, chunk in enumerate(
                        pandas.read_csv(
                            self.array_testtest_file_path, chunksize=self.pd_chunksize)):
                    if self.array_test is None:
                        self.array_test = chunk.copy()  # not simply a reference to it
                    else:
                        self.array_test = pandas.concat(
                            [self.array_test, chunk])
                    del chunk
                    gc.collect()

            else:
                self.array_test = pandas.read_csv(self.test_file_path)

        elif self.use_list:
            csv_iter = csv.reader(open(self.test_file_path, 'rb'))
            self.array_test = [row for row in csv_iter]
            
    

    def process_test(self):
        # array to be returned (as a reference to self.array_test)
        X_test = self.array_test
        encoders_local = self.encoders
        tc = self.target_column if self.target_column != - \
            1 else len(self.encoders) - 1
        # Now encoders local should match test columns after we've popped the
        # encoder for the target column
        encoders_local.pop(tc)
        is_header = True if True in self.header_or_not else False
        adjusted_exclude_columns = []

        is_dt_local = self.col_is_datetime
        is_dt_local.pop(tc)

        for i in self.exclude_columns:
            if i < tc:
                adjusted_exclude_columns.append(i)
            if i > tc:  # Because if the target column was in the middle of the train array, the values provided for
                # excl_cols greater than indexof target col
                # ... need to be reduced by 1 since test array would already lacks the target column
                adjusted_exclude_columns.append(i - 1)
        print(len(encoders_local), len(self.encoders))
        for x in adjusted_exclude_columns:
            encoders_local[x] = 'Dont need this'
            is_dt_local[x] = False

        if isinstance(self.array_test, numpy.ndarray):
            if is_header:
                X_test = self.array_test[1:]
            else:
                q = None  # Completely useless atm. Feel we might need another case here in future

            try:
                rng = xrange(0, len(X_test[0, 0:]))
            except NameError:
                rng = range(0, len(X_test[0, 0:]))

            for column in rng:
                if column in adjusted_exclude_columns:  # no point processing columns we will later exclude
                    continue

                # If column is categorical but also a datetime, don't convert it
                if not isinstance(encoders_local[column], str) and is_dt_local[column] is False:
                    # convert to number labels using LabelEncode
                    # print(column)
                    if self.advanced_ops:  # remove leading or trailing spaces
                        X_test[:, column] = numpy.char.strip(X_test[:, column])
                    # output of encoder.transform is a numpy.ndarray, FYI
                    X_test[:, column] = encoders_local[column].transform(X_test[:, column], True)
                if self.dt_convert == 1:
                    if is_dt_local[column]:
                        self.dt_array_test.append(numpy.array(
                            [parse(i, dayfirst=self.dayfirst) for i in X_test[:, column]]))

            array_of_col_index = [n for n in range(0, len(X_test[0]))]
            # Pick only the columns not listed to be excluded
            X_test = X_test[:, [i for i in array_of_col_index if (
                i not in adjusted_exclude_columns)]]

        if isinstance(self.array_test, pandas.core.frame.DataFrame):
            if is_header:
                X_test = self.array_test[1:]
            else:
                q = None

            # Handle missing values
            if self.missing_vals == 'fill' or self.missing_vals == 'drop':
                # Missing values shouldn't be dropped in the test set
                for index, column in enumerate(self.array_test.columns):
                    if self.col_is_categorical[index]:
                        mode = stats.mode(X_test.loc[:][column])[0][0]
                        X_test[column] = X_test[column].fillna(mode)
                    else:
                        mean = numpy.mean(
                            X_test[column][
                                pandas.notnull(
                                    X_test[column])])
                        X_test[column] = X_test[column].fillna(mean)

            for index, column in enumerate(X_test.columns):
                if index in adjusted_exclude_columns:  # no point processing columns we will later exclude
                    continue

                if not isinstance(
                        encoders_local[index],
                        str) and is_dt_local[index] is False:
                    if self.advanced_ops:
                        X_test[column] = X_test[column].str.strip()
                    # this back references and actually modifies the ooriginal
                    # test.csv in memory
                    X_test.loc[:][column] = encoders_local[
                        index].transform(X_test[column], True)
                # Attach a datetime object for each column.
                if self.dt_convert == 1:
                    if self.col_is_datetime[index]:
                        self.dt_array_test.append(pandas.Series(
                            [parse(i, dayfirst=self.dayfirst) for i in X_test[column]]))

            for i in adjusted_exclude_columns:
                no_use = X_test.pop(i)

        print('len of enc local and dt_local ', len(encoders_local), len(is_dt_local))
        return X_test

    """def drop(self,cols):                        #arg "cols" can be a single index or array in indexes
        if not hasattr(self, "is_processed"):
            raise ValueError("DataWiz array must be processed before dropping columns.")
        drop_indexes = []
        if type(cols)== int:
            drop_indexes.append(cols)
        else:
            drop_indexes = cols """



