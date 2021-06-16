# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 22:30:06 2021

@author: Asus
"""

# Here we will import the requested libraries
import numpy as np
import pandas as pd
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans # Scikit-Learn's algorithm for this exercise

# Now we are goint to read the csv (Wholesale customers data set) with pandas
path = "~/Desktop/KCL/Semester 2/Data Mining/Coursework 1/{}" # Here you put the path to the folder where your csv is stored
df = pd.read_csv(path.format('Wholesale customers data.csv'))
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv') # If you want to read it directly from the Internet

# We have to drop Channel and Region, columns that we will not need for this exercise
df = df.drop(['Channel', 'Region'], axis=1)


# Now let's print some general information about this csv to see how it is
print("This 'Wholesale customers data.csv' has a shape of {} and here is some information about it: \n".format(df.shape))
print(df.info())
print(df.describe())
print('\n')

# 
# CREATE TABLE FOR SOME DESCRIPTIVE INFORMATION
#

# We will print this information helping ourselves with PrettyTable()
table = PrettyTable() # Create table
table.field_names = ['ATTRIBUTE', 'RANGE', 'MEAN'] # Add titles to the columns of our table
for c in df.columns:
    table.add_row([c, [min(df[c]), max(df[c])], np.mean(df[c])]) # We add the attribute, its [range] and its mean
print(table)
print('\n')

# 
# PREVIOUS VISUALISATION OF THE DATA
#

# Let's try to previsualise the data first. With the library seaborn it is quite straightforward
sns.pairplot(data=df, kind='scatter', diag_kind='hist')

# 
# K-MEANS WITH K=3
#

# We already have all our data prepared in  a numerical format, so we can directly apply the algorithm. We do not need to split the 
# data set in train and test because it is an unsupervised learning problem and it is not necessary.
X = df

# Here we initialise the model
kMeans3 = KMeans(n_clusters=3, random_state=42) # n_clusters == k == number of clusters/centroids

# Now we train it
kMeans3.fit(X)

# We already have the possible labels, so let's store them in an array
y3 = kMeans3.labels_ # This would be used later to print the clusters in different colors

# We will also store the centroids in another array to try to print them in the pair plots too
centroids3 = kMeans3.cluster_centers_

# Let's create the pair plots with seaborn and matplotlib.pyplot. We will do so by looping through all the columns
columns = X.columns # Store the column names in an array
count = 1 # to enumerate our plots
for i in range(len(columns)): # Loop through all columns
    for j in range(i+1, len(columns)): # For EACH column, loop through all the other columns that haven't been plotted yet
        plt.scatter(X[y3 == 0].iloc[:, i], X[y3 == 0].iloc[:, j], c='lightsteelblue', alpha=0.6, label='Cluster 1')
        plt.scatter(X[y3 == 1].iloc[:, i], X[y3 == 1].iloc[:, j], c='yellow', alpha=0.6, label='Cluster 2')
        plt.scatter(X[y3 == 2].iloc[:, i], X[y3 == 2].iloc[:, j], c='salmon', alpha=0.6, label='Cluster 3')
        for cent, color in zip(range(centroids3.shape[0]), ['slategrey', 'orange', 'red']):
            plt.scatter(centroids3[cent, i], centroids3[cent, j], marker='*', c=color) # We will plot the centroids as stars of a similar colour
        plt.xlabel(columns[i]) # First attribute name
        plt.ylabel(columns[j]) # Second attribute name
        plt.legend(loc='upper right') # Show legend
        plt.title('Plot #' + str(count) + ': ' + columns[j] + ' VS. ' + columns[i], fontweight='bold') # Show title
        plt.tight_layout()
        plt.show()
        count+=1 # to have some sort of index for each plot
        
# Now we will calculate the different cluster evaluation scores (BC and WC) for k=3
# We will help ourselves with the built-in function np.linalg.norm(a-b), which gives us the euclidean distance between a and b
# We have the BC score in the slides and in the report
def bc(centroids): # We create a function to recycle this code for k=5 and k=10
    bc_score = 0
    for k in range(len(centroids)-1): # For every cluster
        for i in range(k+1, len(centroids)): # For EACH cluster, iterate through all the other clusters
            bc_score += np.linalg.norm(centroids[k, :]- centroids[i, :])**2 # Calculate the distance and sum it to the total distance
    return bc_score
print("The Between Cluster Score for k=3 is: {:.4f}".format(bc(centroids3)))

# Now we have to calculate the Within Cluster Distance for k=3. So, we have to sum for each point of a cluster the distance to the centroid
# of that cluster, and then sum up the three different Within Cluster Distance
def wc(centroids, X, y):
    wc_score = 0
    for k in range(len(centroids)): # for each cluster (k=3)
        for i in X[y == k].index: # for each point of that specific cluster
            wc_score += np.linalg.norm(X.iloc[i, :] - centroids[k, :])**2 # Calculate the distance and sum it to the total distance
    return wc_score
print("The Within Cluster Score for k=3 is: {:.4f}".format(wc(centroids3, X, y3)))
# I found out later that this can be done with kMeans.inertia_. Anyway, I will leave my code here

# The last metric we need is the ratio BC/WC, which is a simple division
print("The Ratio BC/WC for k=3 is: {:.6f}\n".format(bc(centroids3)/wc(centroids3, X, y3)))
        
# 
# K-MEANS WITH K=5
#

# Let's repeat the same process for each k-means model
kMeans5 = KMeans(n_clusters=5, random_state=42) # Here we initialise the model for k=5
kMeans5.fit(X) # we train it
y5 = kMeans5.labels_ # we save the labels of this model
centroids5 = kMeans5.cluster_centers_ # we also store the centroids

# Now we print the metrics we are interested in:
print("The Between Cluster Score for k=5 is: {:.4f}".format(bc(centroids5)))
print("The Within Cluster Score for k=5 is: {:.4f}".format(wc(centroids5, X, y5)))
print("The Ratio BC/WC for k=5 is: {:.6f}\n".format(bc(centroids5)/wc(centroids5, X, y5)))

# 
# K-MEANS WITH K=10
#

# Let's repeat the same process for each k-means model
kMeans10 = KMeans(n_clusters=10, random_state=42) # Here we initialise the model for k=10
kMeans10.fit(X) # we train it
y10 = kMeans10.labels_ # we save the labels of this model
centroids10 = kMeans10.cluster_centers_ # we also store the centroids

# Now we print the metrics we are interested in:
print("The Between Cluster Score for k=10 is: {:.4f}".format(bc(centroids10)))
print("The Within Cluster Score for k= 10 is: {:.4f}".format(wc(centroids10, X, y10)))
print("The Ratio BC/WC for k=10 is: {:.6f}\n".format(bc(centroids10)/wc(centroids10, X, y10)))

# To print it nice with PrettyTable():
table2 = PrettyTable() # Create table
table2.field_names = [' ', 'k=3', 'k=5', 'k=10'] # Add titles to the columns of our table
table2.add_row(['BC', bc(centroids3), bc(centroids5), bc(centroids10)]) # We add the BC scores for each k
table2.add_row(['WC', wc(centroids3, X, y3), wc(centroids5, X, y5), wc(centroids10, X, y10)]) # WC scores for each k
table2.add_row(['BC/WC', bc(centroids3)/wc(centroids3, X, y3), bc(centroids5)/wc(centroids5, X, y5), bc(centroids10)/wc(centroids10, X, y10)]) # Ratio BC/WC for each k
print("The metrics of BC, WC and Ratio BC/WC for different number for k is the following:")
print(table2)

# I'm going to leave here the code to plot the clusters when k=10 which is very similar to when k=3

#count = 1 # to enumerate our plots
#for i in range(len(columns)): # Loop through all columns
#    for j in range(i+1, len(columns)): # For EACH column, loop through all the other columns that haven't been plotted yet
#        plt.scatter(X[y10 == 0].iloc[:, i], X[y10 == 0].iloc[:, j], c='lightsteelblue', alpha=0.6, label='Cluster 1')
#        plt.scatter(X[y10 == 1].iloc[:, i], X[y10 == 1].iloc[:, j], c='yellow', alpha=0.6, label='Cluster 2')
#        plt.scatter(X[y10 == 2].iloc[:, i], X[y10 == 2].iloc[:, j], c='salmon', alpha=0.6, label='Cluster 3')
#        plt.scatter(X[y10 == 3].iloc[:, i], X[y10 == 3].iloc[:, j], c='black', alpha=0.6, label='Cluster 4')
#        plt.scatter(X[y10 == 4].iloc[:, i], X[y10 == 4].iloc[:, j], c='purple', alpha=0.6, label='Cluster 5')
#        plt.scatter(X[y10 == 5].iloc[:, i], X[y10 == 5].iloc[:, j], c='green', alpha=0.6, label='Cluster 6')
#        plt.scatter(X[y10 == 6].iloc[:, i], X[y10 == 6].iloc[:, j], c='lightgreen', alpha=0.6, label='Cluster 7')
#        plt.scatter(X[y10 == 7].iloc[:, i], X[y10 == 7].iloc[:, j], c='blue', alpha=0.6, label='Cluster 8')
#        plt.scatter(X[y10 == 8].iloc[:, i], X[y10 == 8].iloc[:, j], c='lightblue', alpha=0.6, label='Cluster 9')
#        plt.scatter(X[y10 == 9].iloc[:, i], X[y10 == 9].iloc[:, j], c='pink', alpha=0.6, label='Cluster 10')
#        for cent, c in zip(range(centroids10.shape[0]), ['steelblue', 'orange', 'red', 'white', 'black', 'darkgreen', 'green', 'darkblue', 'blue', 'purple']):
#            plt.scatter(centroids10[cent, i], centroids10[cent, j], marker='*', c=c)
#        plt.xlabel(columns[i]) # First attribute name
#        plt.ylabel(columns[j]) # Second attribute name
#        plt.legend(loc='upper right') # Show legend
#        plt.title('Plot #' + str(count) + ': ' + columns[j] + ' VS. ' + columns[i], fontweight='bold') # Show title
#        plt.tight_layout()
#        plt.show()
#        count += 1
