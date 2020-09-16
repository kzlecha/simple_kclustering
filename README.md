# simple_kmeans
COSC 499 - Simple Git Excercise: https://people.ok.ubc.ca/bowenhui/499/gitex_indiv.html

A simple kmeans and kmedian classification algorithm in python

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

The algorithm has an option for feature scaling using either MinMax or Standardization.
