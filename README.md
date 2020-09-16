# simple_kmeans
COSC 499 - Simple Git Excercise: https://people.ok.ubc.ca/bowenhui/499/gitex_indiv.html

A simple KMeans and KMedian classification algorithm in Python3, as well as a scaling feature.

### inputs:
The user must input a data matrix for the algorithm to find groups within.
### outputs:
groupings according to which other datapoints are the closest in nD space.

## Requirements
This code was written in Python3 with pandas and numpy.

It is suggested to run a python3 virtual environment and install the requirements with```pip install -r requirements.txt```

## KMeans
This is an implementation of the KMeans Clustering algorithm in Python. It is used as an unsupervised learning cluster algorithm to determine groups within data.

### Assumptions:
KMeans Clustering assumes the following:
  - variance distribution of each attribute is spherical
  - all variables have same variance
  - probablity of k clusters is the same (same distribution)

### Strengths and Weaknesses
#### Strengths
  - converges quickly (for clustering...)
  - Space Complexity: nk

#### Weaknesses:
  - all clusters are spheres with the same radius
  - can only find local solution
  - will find exactly k clusters even if none, more, or less exist
  
### Implemenation Notes
Distance is calculated using euclidean distance. The current implementation cannot handle missing data.

The algorithm can only fit and predict on data enterred, but will not make predictions based on the fitted model. Such feature would be implemented at a later time.

## KMedian
KMedians is a variant on the KMeans clustering algorithm that is designed to create centroids by the median of its group instead of the mean. In this way, KMedians is more robust to outliers and anomalies than KMeans

### Assumptions, Strengths, and Weaknesses:
Because KMedians is a variant of KMedian, it has the same assumptions, strengths, and weaknesses. However, in comparision to KMeans, KMedian is more robust to outliers and anomalies.
