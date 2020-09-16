from k_clustering import KClustering
from pandas import DataFrame, Series


class KMedians(KClustering):
    '''
    KMedians Clustering Algorithm
    Unsupervised Machine Learning with no underlying statistical model

    KMedians is a variant on the KMeans clustering algorithm that is designed
    to create centroids by the median of its group instead of the mean.
    In this way, KMedians is more robust to outliers and anomalies than KMeans
    
    Assumptions:
        - variance distribution of each attribute is spherical
        - all variables have same variance
        - probablity of k clusters is the same (same distribution)

    Strengths:
        - converges quickly (for clustering...)
        - Space Complexity: nk
        - more robust to outliers and anomalies than KMeans

    Weaknesses:
        - all clusters are spheres with the same radius
        - can only find local solution
        - will find exactly k clusters even if none, more, or less exist
    '''

    def __init__(self, k=4, max_iterations=258):
        super(KClustering)
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, data):
        '''
        runs the KMedians classification algorithm on a dataframe
        ---
        Inputs:
            @param data: pandas dataframe with n observations and p attributes
                         data should be scaled for best results
        ---
        Outputs:
            saves centroids and memberships to object.
            centroids: the center/median of the different groups
            memberships: classification membership list for every observation
                         in the sample. This classification is only a local
                         solutation because centroids are assined randomly
        '''
        centroids, memberships = super()._setup(data)

        # algorithm converges when the memberships and centroids stablize
        converged = False
        iterations = 0
        while(not converged):
            # calculate distance between all data and centers (matricies)

            # create a nxk distance matrix
            distance = DataFrame(index=data.index,
                                 columns=[i for i in range(0, self.k)]
                                 )

            for i in range(0,self.k):
                # calculate the euclidean distance to each centroid
                dist = super()._euclidian_distance(data.values, centroids[i])
                distance.loc[:,i] = dist

            # assign group memberships
            membership = distance.idxmin(axis=1)

            # check exit condition
            converged = self._has_converged(data, centroids, membership)

            iterations += 1
            if iterations == self.max_iterations:
                break

        self.centroids = centroids
        self.membership = membership

    def fit_predict(self, data):
        '''
        runs the KMedians classification algorithm on a dataframe
        ---
        Inputs:
            @param data: pandas dataframe with n observations and p attributes
        ---
        Outputs:
            centroids: the center/median of the different groups
            memberships: classification membership list for every observation
                         in the sample. This classification is only a local
                         solutation because centroids are assined randomly
        '''
        self.fit(data)
        return self.centroids, self.membership

    def _has_converged(self, data, centroids, membership):
        '''
        check to see if the kmedians algorithm has converged
        convergence occurs when the 
        ---
        inputs:
            @param data: pandas dataframe with n observations and p attributes
            @param centroids: numpy 2D array representing datavectors
            @param membership: Series denoting what group each datapoint is in
        ---
        outputs:
            True if the algorithm has converged and centroids do not change
            False otherwise
        '''
        #calcuate new group centroids
        old_centroids = centroids.copy()
        
        # update clusters median
        for i in range(0,self.k):
            # update the centroid's range
            centroids[i] = data.where(membership == i).median(axis=0).values

        # check for convergence
        if (centroids == old_centroids).all():
            return True
        else:
            return False
