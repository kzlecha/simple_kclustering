from pandas import DataFrame, Series
import numpy as np


def kmeans(data, k):
    '''
    runs the KMeans classification algorithm on a dataframe
    ---
    Inputs:
        @param data: a pandas dataframe with n observations and p attributes
        @param k: the desired number of groups.
    ---
    Outputs:
        centroids: the center/mean of the different groups
        memberships: classification membership list for every observation in
            the sample. This classification is only a local solutaion because
            centroids are assined randomly
    ---
    Assumptions:

    '''
    # Select a list of centroids to be the groups
    # implementation will use random observations for centroid initialization
    centroids = np.array(data.sample(n=k, axis=0))
    membership = Series(index=data.index)

    # algorithm converges when the memberships and centroids stop changing
    num = 0
    converged = False
    while(not converged):
        # calculate distance between all data and centers (matricies)

        # create a nxk distance matrix
        distance = DataFrame(index=data.index, columns=[i for i in range(0,k)])

        for i in range(0,k):
            # calculate the euclidean distance to each centroid
            distance.loc[:,i] = np.sqrt(np.sum(np.power(data.values - centroids[i], 2), axis=1))
            print(distance)

        # assign group memberships
        closest_indicies = distance[i].idxmin()
        membership.loc[closest_indicies] = i
        print(membership)

        #calcuate new group centroids
        old_centroids = centroids.copy()
        
        # update clusters median
        centroids = data.mean(axis=0).values

        # check for convergence
        print(np.sum(centroids == old_centroids))
        if np.sum(centroids == old_centroids) == k:
            converged = True
        if num == 5:
            converged = True
        else:
            num += 1
    
    return centroids, memberships


def euclidian_distance(point1, point2):
    '''
    find the euclidean distance between two points
    ---
    inputs:
        @param point1: a Series for a data observation
        @param point2: a Series for a different data observation
    ---
    output:
        the euclidean distance between the two points
    ---
    Assumptions:
        - point1 and point2 should be same dimension
        - point1 and point2 should not contain NaN
    Formula for Euclidean Distance:
        dist = sqrt(sum((a[1]+b[1])^2 + ... + (a[n]+b[n])^2))
    '''
    if len(point1) != len(point2):
        raise ValueException("points should be same dimension")
    
    sum_distance = 0;
    for i in range(0, len(point1)):
        # determine the distance in the current dimension
        distance_part = 0

    return np.sqrt(sum_distance)