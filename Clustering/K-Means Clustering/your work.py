# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
#dataset.iloc is format [lines, columns] so the : for the first parameter is all lines, and here we have annual income and spending score for the columns
X = dataset.iloc[:, [3, 4]].values
# y = dataset.iloc[:, 3].values

#train/test set is not a part of this example

# Feature Scaling is taken care of by the library/class

#using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
#we choose 11 since the 11 bound is not included and we want up to 10 wcss (1 is included)
#max iter is just how many times the kmeans++ (or other init method) is iterated for a single run to help find the optimal centroid location
#n_init is the amount the with different initial centroids the algorithm is run
#so basically the equation will start out with 10 initial centroid 'seeds' and then run the kmeans++ algorithm 300 times for the optimal centroid location
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    #we still need to fit the algorithm to the data X
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
#then plot the data
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#applying k-means to the mall dataset with the correct number of clusters (5) that we found with the elbow method
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
#the x and y params are saying X= the first column of x where y_kmeans == 0 then y = the second column of x where y_kmeans == 0. The X I'm referring to is on line 10, not X in the og dataset
#
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', edgecolors = 'black', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], c = 'blue', edgecolors = 'black', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], c = 'green', edgecolors = 'black', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], c = 'cyan', edgecolors = 'black', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], c = 'magenta', edgecolors = 'black', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', edgecolors = 'black', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
#legend is tied to the labels above, it takes the kwargs on the lines above
plt.legend()
plt.show()