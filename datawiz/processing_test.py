# Authors: Koye Sodipo <broskoye@hotmail.com>
# License: MIT

import csv
import gc
from sklearn import preprocessing
from random import randint
from scipy import stats
from dateutil.parser import parse
import numpy
import pandas

def read_test(test_file_path,pd_chunksize):
    
    use_numpy = False
    use_pandas = True
    use_list= False

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
                        test_file_path, chunksize=pd_chunksize)):
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

    return array_test
        


def process_test(final_train_cols,array_test,numeric_cols,encoders,encoded_cols,encoder_class_mode,dt_array,dt_cols,missing_vals,dt_convert,data_has_ws):
    
   
    dt_array_test = []

    if isinstance(array_test, pandas.core.frame.DataFrame):
        

        # Only proceed with columns used in the training set
        array_test = array_test[final_train_cols]
       
        # Handle missing values
        if missing_vals == 'fill' or missing_vals == 'drop':
            # Missing values shouldn't be dropped in the test set
            for index, column in enumerate(array_test.columns):
                if column not in numeric_cols:
                    mode = stats.mode(array_test.loc[:][column])[0][0]
                    array_test[column] = array_test[column].fillna(mode)
                else:
                    mean = numpy.mean(
                        array_test[column][
                            pandas.notnull(
                                array_test[column])])
                    array_test[column] = array_test[column].fillna(mean)
    #return array_test

        for index, column in enumerate(array_test.columns):
            
            if column in encoded_cols:
                if data_has_ws:
                    array_test[column] = array_test[column].str.strip()
                # this back references and actually modifies the original
                # test.csv in memory
                index_for_encoder = encoded_cols.index(column)
                array_test[column][~array_test[column].isin(encoders[index_for_encoder].classes_)] = encoder_class_mode[index_for_encoder]
                try:
                    array_test[column] = encoders[index_for_encoder].transform(array_test[column], True)
                except:
                    array_test[column] = encoders[index_for_encoder].transform(array_test[column])
                    
                
            # Attach a datetime object for each column.
            if dt_convert:
                if column in dt_cols:
                    dt_array_test.append(pandas.Series(
                        [parse(i, dayfirst=dayfirst) for i in array_test[column]]))

      

    
    return array_test




