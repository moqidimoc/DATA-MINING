# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 10:19:37 2021

@author: Asus
"""

# Here we will import the requested libraries
import numpy as np
import pandas as pd
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder # for later encoding the variables
from sklearn.model_selection import train_test_split #For the different train/test splits
from sklearn import tree
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score #For the 10-times 10 fold CV too
from sklearn import metrics
import statistics # for calculating the most popular value when filling the missing values

# Now we are going to read the csv (Adult data set) with pandas
path = "~/Desktop/KCL/Semester 2/Data Mining/Coursework 1/{}" # Here you put the path to the folder where your csv is stored
df = pd.read_csv(path.format('adult.csv'))
# pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data') # If you want to read it directly from the Internet

# In this array we will put all the targets (1 if it has an income above 50K, 0 if it's lower)
y = np.array([0 if x == True else 1 for x in df['class'] == '<=50K'])
targets = ['<=50K', '>50K'] # We will use this later

# And now we can drop the columns that the problem formulation demands: 'fnlwgt'
df = df.drop(['fnlwgt'], axis=1)

# Now let's print some general information about this csv to see how it is
print("This 'adult.csv' has a shape of {} and here is some information about it: \n".format(df.shape))
print(df.info())
print(df.describe())
print('\n')

# 
# CREATE TABLE FOR SOME DESCRIPTIVE INFORMATION
#

num_of_instances = len(df) 
num_of_mv = df.isna().sum().sum() # number of missing values
fraction_mv_attributes =  num_of_mv/df.size # first fraction of missing values over all attribute values
num_instances_with_mv = (df.isna().sum(axis=1) > 0).sum() # number of instances with missing values
fraction_instancesmv_allinstances = num_instances_with_mv/num_of_instances # last fraction: instances with missing values
                                                                           # over all instances
table = PrettyTable() # Create table
table.field_names = ['QUESTION', 'ANSWER'] # Add titles of the columns
table.add_row(['number of instances', num_of_instances]) # Add rows to the table
table.add_row(['number of missing values', num_of_mv]) 
table.add_row(['fraction of missing values over all attribute values', fraction_mv_attributes]) 
table.add_row(['number of instances with missing values', num_instances_with_mv]) 
table.add_row(['fraction of instances with missing values over all instances', fraction_instancesmv_allinstances])      
print(table) # Print our the table                                                                    
                                                                   

# And now we can drop the columns that we aren't going to need
df = df.drop(['class', 'education'], axis=1)
        
#
# ENCODE THE CATEGORICAL FEATURES OF THE DATA SET
#

# First, we initialise the OrdinalEncoder
encoder = OrdinalEncoder()
    
# Now we apply the encoding to our Dataframe with the data (df)
# As the Decision Tree is going to predict without missing values, we will drop them in this step:
X1 = pd.DataFrame(encoder.fit_transform(df.dropna(how='any')), columns=df.columns) # Here we drop the rows with 'any' missing value
y1 = y[~(df.isna().sum(axis=1) > 0)] # We drop the same instances from the target array

# Lastly, we print the set of possible all possible columns with PrettyTable
print("\nThe encoding of our data set has transformed our attributes in the following manner: ")
table2 = PrettyTable()
table2.field_names = ['ATTRIBUTE', 'SET OF POSSIBLE VALUES']
for c in X1.columns:
    table2.add_row([c, np.sort(X1[c].unique())])
print(table2)

#
# INITIAL DECISION TREE CLASSIFIER (IGNORING MISSING VALUES)
#

# First of all, as we already have our data set prepared for a model, we will split it into train and test to later evaluate
# our model performance on the test set. 
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=0) # Train/Test split

# Now we are going to perform our first model of the coursework. We will do so by realising a 10-times 10-fold . 
# cross-validation. We will start by intialising the Decision Tree Classifier and the CV method:
dtree = tree.DecisionTreeClassifier(random_state=42) # Decision Tree
kcv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=167) # CV method

# Now we train the model: 
dtree = dtree.fit(X1_train, y1_train)

# And we validate our model:
scores = cross_val_score(dtree, X1_train, y1_train, scoring='accuracy', cv=kcv, n_jobs=-1)

# Accuracy: The number of correct predictions made divided by the total number of predictions made.
print('\nTrain Error Rate for the whole dataset: {:.3f} ({:.3f})'.format(1-np.mean(scores), np.std(scores))) # Here we print the scores of our training model

# Now let's see how well it performs on the test set: 
y1_hat = dtree.predict(X1_test)

print("Test Error Rate for the whole dataset: {:.3f}".format(1-metrics.accuracy_score(y1_test, y1_hat)))
    
# Now, let's compute a confusion matrix. We will create a function for this:
def confusion(y_test, y_hat, title):
    cm = metrics.confusion_matrix(y_test, y_hat)
    
    # From https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
    ax= plt.subplot()
    sns.heatmap(cm, cmap='Greens', annot=True, ax = ax) #annot=True to annotate cells
    
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('Actual labels')
    ax.set_title("Confusion Matrix for {}".format(title))
    ax.xaxis.set_ticklabels(targets)
    ax.yaxis.set_ticklabels(targets)
    plt.show()
    
    # We can also compute the precision (TP/TP+FP), should be (13/13, 15/15, 9/10)
    precision = metrics.precision_score(y_test, y_hat, average=None)
    print("Precision score for {}= TP/(TP+FP) =".format(title))
    for i in range(len(precision)):
        print("\t{}: {:.3f}".format(targets[i], precision[i]))
        
    # And we can do the same with the recall (TP/TP+FN), should be (13/13, 15/16, 9/9)
    recall = metrics.recall_score(y_test, y_hat, average=None)
    print("Recall score for {}= TP/(TP+FP) =".format(title))
    for i in range(len(recall)):
        print("\t{}: {:.3f}".format(targets[i], recall[i]))
        
    # Lastly, you can also compute the F1 score (2TP/(2TP+FP+FN))
    f1 = metrics.f1_score(y_test, y_hat, average=None)
    print("f1 score for {}= 2*(precision*recall)/(precision+recall) =".format(title))
    for i in range(len(f1)):
        print("\t{}: {:.3f}".format(targets[i], f1[i]))
    
    print('\n')

confusion(y1_test, y1_hat, 'the whole dataset')

#
# DECISION TREES CLASSIFIERS (2 BASIC APPROACHES FOR HANDLING MISSING VALUES)
#

# Let's start by merging X and y so we have the true targets with its respective instance
df = pd.concat([df, pd.DataFrame(y)], axis=1)

# Now we will create a new df with the exemplars with missing values:
X21 = df[(df.isna().sum(axis=1) > 0)]

# We can already drop these values from the initial data set (df)
df_without_nans = df.dropna(how='any')

# We select 3620 random instances from df to store them in X22, the second group of our final data set
X22 = df_without_nans.sample(3620)

# We can finally concat them and end up with X2==D'. We add .sample(frac=1) after to shuffle our instances and so the missing values
# aren't on top and the other instances below.
X2 = pd.concat([X21, X22]).sample(frac=1)

# We have to remember to separate the targets from the data set to have them in another array
y2 = X2.pop(0) # to have the targets stored in another array

# Next step: create D1' by replacing the missing values with a new value called 'missing'
D1 = X2.fillna('missing')

# Last step: create D2' by replacing the missing values with the most popular value for each attribute
filling = {
    'workclass': statistics.mode(df['workclass']),
    'occupation': statistics.mode(df['occupation']),
    'native-country': statistics.mode(df['native-country']),
    } # dictionary used to replace the missing values from X2
D2 = X2.fillna(filling) # D2' with the missing values replaced with the mode of each colu"mn

# Now we will do the same steps for each dataframe recycling the functions we created before

# Step 1: encoding
D1 = pd.DataFrame(encoder.fit_transform(D1), columns=np.delete(df.columns, len(df.columns)-1)) # we delete the last element of the columns (the targets)
D2 = pd.DataFrame(encoder.fit_transform(D2), columns=np.delete(df.columns, len(df.columns)-1)) # we delete the last element of the columns (the targets)

# Step 2: training our models

    # D1'
dtree1 = tree.DecisionTreeClassifier(random_state=42) # Decision Tree
dtree1 = dtree1.fit(D1, y2) # train the model with the targets we dropped before from X2
scores1 = cross_val_score(dtree1, D1, y2, scoring='accuracy', cv=kcv, n_jobs=-1) # model validation
print("Train Error Rate for D1': {:.3f} ({:.3f})".format(1-np.mean(scores1), np.std(scores1))) # scores of our training model

    # D2'
dtree2 = tree.DecisionTreeClassifier(random_state=42) # Decision Tree
dtree2 = dtree2.fit(D2, y2) # train the model with the targets we dropped before from X2
scores2 = cross_val_score(dtree2, D2, y2, scoring='accuracy', cv=kcv, n_jobs=-1) # model validation
print("Train Error Rate for D2': {:.3f} ({:.3f})\n".format(1-np.mean(scores2), np.std(scores2))) # scores of our training model

# Step 3: modify the initial data set so it is accepted by our model

testdf = df.drop(X2.index) # We drop the instances that belonged to the training set from our test set
y_test = testdf.pop(0) # We store the actual labels of the test set instances in this variable

    # D1'
testD1 = testdf.fillna('missing') # fill missing values with 'missing'
testD1 = pd.DataFrame(encoder.fit_transform(testD1), columns=testdf.columns) # encode the test data set

    # D2'
testD2 = testdf.fillna(filling) # fill missing values with the most popular value (stored in filling)
testD2 = pd.DataFrame(encoder.fit_transform(testD2), columns=testdf.columns) # encode the test data set

# Step 4: Evaluate our models with the train set, which in this case is the whole dataset D, what we called before X

    # D1': we have to remember to replace the missing values with 

yd1_hat = dtree1.predict(testD1) # predictions
print("Test Error rate for D1': {:.3f}".format(1-metrics.accuracy_score(y_test, yd1_hat))) # Accuracy score for D1''s model
confusion(y_test, yd1_hat, "D1'") # confusion matrix

    # D2'
yd2_hat = dtree2.predict(testD2) # predictions
print("Test Error rate for D2': {:.3f}".format(1-metrics.accuracy_score(y_test, yd2_hat))) # Accuracy score for D2''s model
confusion(y_test, yd2_hat, "D2'") # confusion matrix