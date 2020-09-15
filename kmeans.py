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
    converged = False
    while(not converged):
        # calculate distance between all data and centers (matricies)

        # create a nxk distance matrix
        distance = DataFrame(index=data.index, columns=[i for i in range(0,k)])

        for i in range(0,k):
            # calculate the euclidean distance to each centroid
            distance[i] = np.sqrt(np.sum(np.power(data[i,:] - centroids[i], 2), axis=1))

            # assign group memberships
            closest_indicies = distance[i].idxmin()
            membership.loc[closest_indicies] = i

        #calcuate new group centroids
        old_centroids = centroids.copy()
        
        # update clusters median
        centroids = data.mean(axis=0).values

        # check for convergence
        if np.sum(centroids == old_centroids) == k:
            converged = True        
    
    return centroids, memberships

