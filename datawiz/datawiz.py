# Authors: Koye Sodipo <broskoye@hotmail.com>
# License: MIT


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


class datawiz:
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
        file_path = train_path
        test_file_path = test_path
        to_use = use
        use_numpy = True
        use_pandas = False
        use_list = False
        target_column = target_col
        exclude_columns = exclude_cols
        test_split = 0.2
        missing_vals = missing_values
        pd_chunksize = pds_chunksize
        dt_convert = dt_convert

        # Removes white space in string columns, datetime conversion
        advanced_ops = advanced_ops
        # Specifies whether recommended columns to be dropped are actually
        # dropped autommatically
        drop_cols = drop_cols

        # Advanced Defult settings (not editable through arguments)

        # should the date parser consider the first number group ('09') in '09/12/2010' as the day?
        dayfirst = True

        array = []
        array_test = []
        ans = -1
        accum = []
        header_or_not = []
        col_is_categorical = []
        col_is_datetime = []
        encoders = []
        header = []

        dt_array = []
        dt_array_test = []

        while True:
            try:
                if file_path is None:
                    file_path = input(
                        'Enter train file path (surround with quotes)   :')
                if to_use is None:
                    to_use = input(
                        'Enter 0 for numpy, 1 for pandas and 2 for list:  ')
                if target_column == -99:
                    target_column = input(
                        'Enter index of the the target column. Enter "None" if no target:  ')
                if exclude_columns == []:
                    while True:
                        excl = input(
                            'List the index of columns to exclude. Enter "None" to quit...')
                        if excl is None:
                            break
                        exclude_columns.append(excl)

                ans = 1
            except:
                raise NameError('Please enter valid answers')

            if ans == 1 or ans == 0:
                break

        use_numpy = True if (
            to_use == 0 or to_use == 'numpy') else False
        use_pandas = True if (
            to_use == 1 or to_use == 'pandas') else False
        use_list = True if (
            to_use == 2 or to_use == 'list') else False

        if use_numpy:
            csv_iter = csv.reader(open(file_path, 'rb'))
            data = [row for row in csv_iter]
            array = numpy.array(data)
            del data
            gc.collect()

        elif use_pandas:
            if pd_chunksize > 0:
                array = None
                for i, chunk in enumerate(pandas.read_csv(
                        file_path, chunksize=pd_chunksize, low_memory=False)):
                    if array is None:
                        array = chunk.copy()  # not simply a reference to it
                    else:
                        array = pandas.concat([array, chunk])
                    del chunk
                    gc.collect()

            else:
                array = pandas.read_csv(file_path)

        elif use_list:
            csv_iter = csv.reader(open(file_path, 'rb'))
            array = [row for row in csv_iter]
            


def read_test(self):
    
    while True:
        ans_t = 1
        try:
            if test_file_path is None:
                test_file_path = input(
                    'Enter file path (surround with quotes)   :')
            break
        except:
            raise NameError

    # User can reset these by manipulating the class objects directly
    # before calling read_test()
    if use_numpy:
        csv_iter = csv.reader(open(test_file_path, 'rb'))
        data = [row for row in csv_iter]
        array_test = numpy.array(data)
        del data
        gc.collect()

    elif use_pandas:
        if pd_chunksize > 0:
            array_test = None
            for i, chunk in enumerate(
                    pandas.read_csv(
                        array_testtest_file_path, chunksize=pd_chunksize)):
                if array_test is None:
                    array_test = chunk.copy()  # not simply a reference to it
                else:
                    array_test = pandas.concat(
                        [array_test, chunk])
                del chunk
                gc.collect()

        else:
            array_test = pandas.read_csv(test_file_path)

    elif use_list:
        csv_iter = csv.reader(open(test_file_path, 'rb'))
        array_test = [row for row in csv_iter]
        


def process_test(self):
    # array to be returned (as a reference to array_test)
    X_test = array_test
    encoders_local = encoders
    tc = target_column if target_column != - \
        1 else len(encoders) - 1
    # Now encoders local should match test columns after we've popped the
    # encoder for the target column
    encoders_local.pop(tc)
    is_header = True if True in header_or_not else False
    adjusted_exclude_columns = []

    is_dt_local = col_is_datetime
    is_dt_local.pop(tc)

    for i in exclude_columns:
        if i < tc:
            adjusted_exclude_columns.append(i)
        if i > tc:  # Because if the target column was in the middle of the train array, the values provided for
            # excl_cols greater than indexof target col
            # ... need to be reduced by 1 since test array would already lacks the target column
            adjusted_exclude_columns.append(i - 1)
    print(len(encoders_local), len(encoders))
    for x in adjusted_exclude_columns:
        encoders_local[x] = 'Dont need this'
        is_dt_local[x] = False

    if isinstance(array_test, numpy.ndarray):
        if is_header:
            X_test = array_test[1:]
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
                if advanced_ops:  # remove leading or trailing spaces
                    X_test[:, column] = numpy.char.strip(X_test[:, column])
                # output of encoder.transform is a numpy.ndarray, FYI
                X_test[:, column] = encoders_local[column].transform(X_test[:, column], True)
            if dt_convert == 1:
                if is_dt_local[column]:
                    dt_array_test.append(numpy.array(
                        [parse(i, dayfirst=dayfirst) for i in X_test[:, column]]))

        array_of_col_index = [n for n in range(0, len(X_test[0]))]
        # Pick only the columns not listed to be excluded
        X_test = X_test[:, [i for i in array_of_col_index if (
            i not in adjusted_exclude_columns)]]

    if isinstance(array_test, pandas.core.frame.DataFrame):
        if is_header:
            X_test = array_test[1:]
        else:
            q = None

        # Handle missing values
        if missing_vals == 'fill' or missing_vals == 'drop':
            # Missing values shouldn't be dropped in the test set
            for index, column in enumerate(array_test.columns):
                if col_is_categorical[index]:
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
                if advanced_ops:
                    X_test[column] = X_test[column].str.strip()
                # this back references and actually modifies the ooriginal
                # test.csv in memory
                X_test.loc[:][column] = encoders_local[
                    index].transform(X_test[column], True)
            # Attach a datetime object for each column.
            if dt_convert == 1:
                if col_is_datetime[index]:
                    dt_array_test.append(pandas.Series(
                        [parse(i, dayfirst=dayfirst) for i in X_test[column]]))

        for i in adjusted_exclude_columns:
            no_use = X_test.pop(i)

    print('len of enc local and dt_local ', len(encoders_local), len(is_dt_local))
    return X_test

"""def drop(self,cols):                        #arg "cols" can be a single index or array in indexes
    if not hasattr(self, "is_processed"):
        raise ValueError("datawiz array must be processed before dropping columns.")
    drop_indexes = []
    if type(cols)== int:
        drop_indexes.append(cols)
    else:
        drop_indexes = cols """



