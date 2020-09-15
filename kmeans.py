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
            distance.loc[:,i] = np.sqrt(np.sum(np.power(data.values - centroids[i], 2), axis=1))

        # assign group memberships
        membership = distance.idxmin(axis=1)

        #calcuate new group centroids
        old_centroids = centroids.copy()
        
        # update clusters median
        for i in range(0,k):
            # update the centroid's range
            centroids[i] = data.where(membership == i).mean(axis=0, skipna=True).values

        # check for convergence
        if np.sum(centroids == old_centroids, axis=1).all():
            converged = True
    
    return centroids, membership
