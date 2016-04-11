# Clustering
This project aims at finding meaningful clusters in short texts. Each short text is read as a separate document, then a set of pre processing steps are done on these short texts. Once, we are done with the NLP tasks, we use a popular machine learning algorithm -  K Means(You can learn more on K Means at https://datasciencegeeks.net/2016/03/16/understanding-k-means-clustering/) to cluster these documents into meaningful categories/clusters.

The Clustering.py file takes in input a single document containing short texts. We assume that each short text is on a single line in the text file. The Clustering.py will then print the clusters for the short texts in the order of the short text. Also, for every cluster, we are printing two answers that are closest to the centroids and two that are farthest from the centroids. For calculating distance, we use euclidean distance between the two vectors.

