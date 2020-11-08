# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 11:25:29 2020

@author: sreeja
"""
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#read data set
iris = pd.read_csv('Iris.csv')
data_set = iris.iloc[:, [1, 2, 3, 4]].values

#predicting the optimum number of clusters using KMeans clustering technique
from sklearn.cluster import KMeans
#declare an empty list to save the inertia values
inertia = []

for i in range(1,10):
    kmeans = KMeans(n_clusters=i,random_state=1234)
    kmeans.fit(data_set)
    inertia.append(kmeans.inertia_)

    
#from Inertia values we get optimum number of cluster is 3 after which the values 
#doesn't decrease significantly with every iteration.(Elbow Method)
#apply KMeans algorithm and find the clusters

kmeans = KMeans(n_clusters=3,random_state=1234)
kmeans.fit(data_set)

#using label_ we can find to which cluster our data points are assigned
labels = kmeans.labels_
#visualizing the clusters
for i in range(0, data_set.shape[0]):
    if labels[i] == 0:
        cluster1 = plt.scatter(data_set[i,0],data_set[i,1],c='r') 
    elif labels[i] == 1:
        cluster2 = plt.scatter(data_set[i,0],data_set[i,1],c='g') 
    elif labels[i] == 2:
        cluster3 = plt.scatter(data_set[i,0],data_set[i,1],c='b') 
        
#plotting the centroids of the clusters
center = plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'y')

plt.legend([cluster1, cluster2, cluster3,center],['Cluster 0', 'Cluster 1','Cluster 2','Centroids'])
plt.show()

