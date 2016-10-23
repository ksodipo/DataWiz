# Authors: Koye Sodipo <broskoye@hotmail.com>
# License: BSD 3 clause

import numpy
import pandas
import csv
import gc
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.tree import export_graphviz
from sklearn import preprocessing
from random import randint
from scipy import stats
#Need to take ownership of the read process. Read either in numpy or pandas. Reading in lists is inefficient.
#Need memory efficient way of converting list to numpy/pandas array

#Two fundamental problems with determining whether 1st row is header:
        # 1.) If all  elements (including the header) are numbers, code will think header is just any other data point
        # 2.) If all elements (including the header) are strings, code will think header is just any othe data point. This can be solved if we assume the header title is unique to the entire column (i.e. occurs only once as header). Might revisit later. CBA atm.
#Need to figure out how to address missing values. Persistent bug.




class DataWiz:
    def __init__( self,train_path=None,test_path=None,use=0, target_col=None,exclude_cols=[],missing_values='fill',pds_chunksize=0):
        
        #Default settings
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
        #Advanced Defult settings
        self.pd_chunksize = pds_chunksize

        self.array = []
        self.array_test = []
        ans = -1
        self.accum = []
        self.header_or_not = []
        self.col_is_categorical = []
        self.encoders = []
        self.header = []
        
        while(True):
                try:
                        if self.file_path==None:
                                file_path = input('Enter file path (surround with quotes)   :')
                        if self.to_use == None:
                            self.to_use =  input('Enter 0 for numpy, 1 for pandas and 2 for list:  ')
                        if self.target_column == None:
                            self.target_column = input('Enter index of the the target column:  ')
                        if self.exclude_columns == []:
                            while (True):
                                    excl = input('List the index of columns to exclude. Enter "None" to quit...')
                                    if excl==None:
                                            break
                                    self.exclude_columns.append(excl)
                                
                        ans = 1
                except:
                        NameError
                        print 'Please enter valid answers'
                        
                if ans == 1 or ans == 0:
                        break
                
        self.use_numpy = True if self.to_use == 0 else False
        self.use_pandas = True if self.to_use == 1 else False
        self.use_list = True if self.to_use == 2 else False

        if self.use_numpy:
                csv_iter = csv.reader(open(self.file_path, 'rb'))
                data = [row for row in csv_iter]
                self.array = numpy.array(data)
                del data
                gc.collect()
                
        elif self.use_pandas:
                if self.pd_chunksize>0:
                        self.array = None
                        for i, chunk in enumerate(  pandas.read_csv(self.file_path,chunksize=self.pd_chunksize)  ):
                                if self.array is None:
                                        self.array = chunk.copy()   #not simply a reference to it
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
        
        if type(self.array) == numpy.ndarray:
            
            for column in xrange(0,len(self.array[0,0:])):                              #Test each column
                test_value_types = [ randint(1,len(self.array)-1) for i in xrange(0,41) ] #initialize an array of 40 valid indexes to randomly sample in a given column. We reset the first value of the array to 0 to test the potential header row
                test_value_types[0] = 0
                self.accum = []                                                            #assumes labels are not integers
                for index in test_value_types:
                    try:
                        float(self.array[0:,column][index])                             #Better to use float() than int() as int('123.45') will throw a ValueError, giving the impression that we're dealing with a string
                        if self.array[0][column]=='Fare':
                                print 'Hit',index,self.array[0:,column][index]
                        self.accum.append(1)
                    except:
                        ValueError
                        if self.array[0][column]=='Fare':
                                print 'Miss',index,self.array[0:,column][index]
                        self.accum.append(0)

                if type(self.array[0,column]) == numpy.string_ and sum(self.accum)<41 and sum(self.accum)>0:       
                    self.header_or_not.append(True)                                      #This logic fails though, if the entire dataset is made of categorical strings and has NO headers. It will still assume 1st item is header regardless.
                else:
                    self.header_or_not.append(False)


                if sum(self.accum)<35:                                   #if the sum of 1s (instances where we found a number) is less than 35, it's probably a categorical column
                        self.col_is_categorical.append(True)             #we might have dirty e.g. nan or a one off string  or missing data which would trick the code into thinking the column is categorical. 
                else:
                        self.col_is_categorical.append(False)


                    
            is_header = True if True in self.header_or_not else False           #Here we decide whether or not the data has headers

            if (is_header):
                #convert the numpy columns that were incorrectly assumed to be strings (and are numbers) to numbers... Actually, this isn't necessary as the sklearn DT converts all strings to floats
                #if header, split header from data. Then detect categorical columns. create label encoder for that
                    self.header = self.array[0]
                    #print 'Header Row: ', self.header
                    
                    ndata = self.array[1:]
            else:
                    ndata = self.array[0:]
                    self.header = [ str(i) for i in xrange(0,len(self.array[0])) ] #Make the header array out of indexes


            for column in xrange(0,len(self.array[0,0:])):
                    if (self.col_is_categorical[column]):
                            #convert to number labes using LabelEncode
                            encoder = preprocessing.LabelEncoder()
                            encoder.fit(ndata[:,column])
                            no_of_unique = len(encoder.classes_)
                            if float(no_of_unique)/float(len(self.array)) > 0.25:                       #if we have so many unique labels relative to the number of rows, it's probably a useless feature or an identifier (both usually) e.g. a name, ticket number, phone number, staff ID. More feature engineering usually required. Unsuprvised PCA perhaps.
                                    encoder = 'Column propably not useful'                                                          #... also, even if we accidentally rule out a legitimate feature, the metric being > 0.25  would probably be a feature that'll cause overfitting
                                    self.encoders.append(encoder)
                                    print 'Consider dropping the column  ',self.header[column], float(no_of_unique),float(len(self.array))
                            else:
                                    ndata[:,column] = encoder.transform(ndata[:,column]) #this back references and actually modifies array
                                    self.encoders.append(encoder)                                    #output of encoder.transform is a numpy.ndarray, FYI

                    else:
                             self.encoders.append('Not a Category')
                             
            Y = ndata[:,self.target_column]
            if self.target_column == -1:
                    self.target_column = len(self.array[0,0:])-1                                                #The extractor wouldn't recognize -1 two lines from here.
            array_of_col_index = [ n for n in xrange(0,len(self.array[0,0:])) ]                                       #get a list of all valid indexes of columns
            X = ndata[ :,[i for i in array_of_col_index if (i!= self.target_column and i not in self.exclude_columns) ] ]                        #this way, we only extract the train columns
                            
                    
    #########################################################################################
        if type(self.array) == pandas.core.frame.DataFrame:
            for column in xrange(0,len(self.array.columns)):                              #Test each column
                test_value_types = [ randint(1,len(self.array)-1) for i in xrange(0,41) ] #initialize an array of 40 valid indexes to randomly sample in a given column. We reset the first value of the array to 0 to test the potential header row
                test_value_types[0] = 0
                self.accum = []                                                            #assumes labels are not integers
                for index in test_value_types:
                    try:
                        float(self.array.loc[index][column])
                        self.accum.append(1)
                    except:
                        ValueError
                        self.accum.append(0)

                if type(self.array.loc[0][column]) == str and sum(self.accum)<41 and sum(self.accum)>0:       #if first item in row is a string and the rest are numbers (i.e. sum of accum falls short of 40), assume that's a header. 
                    self.header_or_not.append(True)                                      #This logic fails though, if the entire dataset is made of categorical strings and has NO headers. It will still assume 1st item is header regardless.
                else:
                    self.header_or_not.append(False)


                if sum(self.accum)<35:                                   #if the sum of 1s (instances where we found a number) is less than 35, it's probably a categorical column
                        self.col_is_categorical.append(True)

                else:
                        self.col_is_categorical.append(False)


                    
            is_header = True if True in self.header_or_not else False           #Here we decide whether or not the data has headers

            if (is_header):
                #convert the pandas columns that were incorrectly assumed to be strings (and are numbers) to numbers... Actually, this isn't necessary as the sklearn DT converts all strings to floats
                #if header, split header from data. Then detect categorical columns. create label encoder for that
                    self.header = self.array[0:1]
                    #print 'Header Row: ',self.header
                    
                    ndata = self.array[1:]
            else:
                    ndata = self.array[0:]

            #Handle missing values  
            
            if self.missing_vals == 'fill':
                for index,column in enumerate(self.array.columns):
                    mode = stats.mode(ndata.loc[:][column])[0][0]
                    ndata[column] = ndata[column].fillna(mode)
                                          
            elif: self.missing_vals == 'drop':
                ndata = ndata.dropna('rows')
            

            for index,column in enumerate(self.array.columns):
                    if (self.col_is_categorical[index]):
                            #convert to number labes using LabelEncode
                            encoder = preprocessing.LabelEncoder()
                            encoder.fit(ndata[column])
                            no_of_unique = len(encoder.classes_)
                            if float(no_of_unique)/float(len(self.array[column])) > 0.25:                       #if we have so many unique labels relative to the number of rows, it's probably a useless feature or an identifier (both usually) e.g. a name, ticket number, phone number, staff ID. More feature engineering usually required. Unsuprvised PCA perhaps.
                                    encoder = 'Category Not Relevant'                                                          #... also, even if we accidentally rule out a legitimate feature, the metric being > 0.25  would probably be a feature that'll cause overfitting
                                    print 'Consider dropping the column ',self.array.columns[index]
                            else:
                                    ndata.loc[:][column] = encoder.transform(ndata[column]) #this back references and actually modifies array           
                            self.encoders.append(encoder)                                #output of encoder.transform is a numpy.ndarray, FYI
                            # In test test, be sure to only transform where col_is_categorical AND encoder != None i.e. 1st instance of True in col_is_categorical checks ast index of encoders array. 2nd checkeck 2nd etc..
                    else:
                             self.encoders.append('Not a Category')
            
            col_names_excl = []                                                         #Get the pandas names of columns before removing target col. 1. to preserve index. 2. Pandas doesn't like dealing with indexes. Prefers names
            for ind in self.exclude_columns:
                col_names_excl.append(ndata.columns[ind])
                
            if self.target_column == -1:
                    self.target_column = len(ndata.columns)-1                                         # .pop sometimes can't deal with -1 as an index
            Y = ndata.pop(self.array.columns[self.target_column])

            
            for i in col_names_excl:
                    garbage = ndata.pop(i)
            del garbage
            gc.collect()
            X = ndata
            
          
        return X,Y    #This is great because X is only a reference to the array object created outside of the function
                        #Our previous setting of ndata to an index of array persists as a global rule. If global array modified out of func, ndata, X changes too.


    def read_test(self):
            
            while(True):
                    ans_t = 1
                    try:
                            if self.test_file_path==None:
                                    self.test_file_path = input('Enter file path (surround with quotes)   :')
                            break
                            
                    except:
                            NameError
                    
            #User can reset these by manipulating the class objects directly before calling read_test()
            if self.use_numpy==True:
                    csv_iter = csv.reader(open(self.test_file_path, 'rb'))
                    data = [row for row in csv_iter]
                    self.array_test = numpy.array(data)
                    del data
                    gc.collect()
                    
            elif self.use_pandas==True:
                    if self.pd_chunksize>0:
                            self.array_test = None
                            for i, chunk in enumerate(  pandas.read_csv(self.array_testtest_file_path,chunksize=self.pd_chunksize)  ):
                                    if self.array_test is None:
                                            self.array_test = chunk.copy()   #not simply a reference to it
                                    else:
                                            self.array_test = pandas.concat([self.array_test, chunk])
                                    del chunk
                                    gc.collect()
                            
                    else:
                            self.array_test = pandas.read_csv(self.test_file_path)

            elif self.use_list==True:
                    csv_iter = csv.reader(open(self.test_file_path, 'rb'))
                    self.array_test = [row for row in csv_iter]

                    

                    
    def process_test(self):
                X_test = self.array_test #array to be returned (as a reference to self.array_test)
                encoders_local = self.encoders
                tc = self.target_column if self.target_column!=-1 else len(self.encoders)-1
                encoders_local.pop(tc)                              #Now encoders local should match test columns after we've popped the encoder for the target column
                is_header = True if True in self.header_or_not else False
                adjusted_exclude_columns = []
                for i in self.exclude_columns:
                            if i<tc:
                                    adjusted_exclude_columns.append(i)
                            if i>tc:                                        #Because if the target column was in the middle of the train array, the values provided for excl_cols greater than indexof target col 
                                    adjusted_exclude_columns.append(i-1)    #... need to be reduced by 1 since test array would already lacks the target column
                print len(encoders_local), len(self.encoders)
                for x in adjusted_exclude_columns:
                        encoders_local[x] = 'Dont need this'
                        
                if type(self.array_test) == numpy.ndarray:
                    
                    if (is_header):
                            X_test = self.array_test[1:]
                    else:
                            q = None                #Completly useless atm. Feel we might need another case here in future
                            
                    
                    for column in xrange(0,len(X_test[0,0:])):
                            
                            if (type(encoders_local[column]) != str):
                                    #convert to number labes using LabelEncode
                                    print column
                                    X_test[:,column] = encoders_local[column].transform(X_test[:,column],True)                                 #output of encoder.transform is a numpy.ndarray, FYI

                    array_of_col_index = [ n for n in xrange(0,len(X_test[0])) ]                                                          
                    X_test = X_test[ :,[i for i in array_of_col_index if (i not in adjusted_exclude_columns) ] ]                           
                    

                if type(self.array_test) == pandas.core.frame.DataFrame:
                    if (is_header):
                            X_test = self.array_test[1:]
                    else:
                            q = None

                    #Handle missing values
                    if (self.missing_vals == 'fill' or self.missing_vals == 'drop' :                #Missing values shouldn't be dropped in the test set
                        for index,column in enumerate(self.array_test.columns):
                            mode = stats.mode(X_test.loc[:][column])[0][0]
                            X_test[column] = X_test[column].fillna(mode)
                                          
                    
                    
                    for index,column in enumerate(X_test.columns):
                            if type(encoders_local[index]) != str:
                                    X_test.loc[:][column] = encoders_local[index].transform(X_test[column],True) #this back references and actually modifies the ooriginal test.csv in memory

                    for i in adjusted_exclude_columns:
                            no_use = X_test.pop(i)

                    
                  
                return X_test
