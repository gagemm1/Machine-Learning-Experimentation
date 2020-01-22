# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
# y = dataset.iloc[:, 3].values

#using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
#linkage is the actual algorithm to create the dendrogram
#have to tell the linkage method which data to use, in this case the X dataset from above
#we're using the ward method because this method tries to minimize the variance within the clusters
#the dendrogram method requires at least the z parameter, the data it's to use for the dendrogram, basically saying create a dendrogram with this data
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
"""
The dendrogram showed above that 5 is the optimal # of clusters. This is because the longest vertical line without an interruption
from a horizontal line anywhere is either of the blue lines on the right, which are by the way interrupted by the green horizontal
lines on the left but the portion on the bottom is the longest. We draw a horizontal line in that area and it intersects 5 vertical
lines, which translates to 5 clusters for optimal classification
"""



# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
#n_clusters is what we found with the dendrogram, 5 clusters
#affinity is what's is used in the k-means algorithm to compute the distance between data points, linkage is what is used to gather/form the clusters. 
"""
The main observations to make are:

single linkage is fast, and can perform well on non-globular data, but it performs poorly in the presence of noise.
average and complete linkage perform well on cleanly separated globular clusters, but have mixed results otherwise.
Ward is the most effective method for noisy data.
"""
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()