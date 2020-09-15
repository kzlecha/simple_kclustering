from pandas import DataFrame, Series
from numpy import array, power, sqrt
from numpy import sum as np_sum


class KMeans:
    '''
    KMeans Clustering Algorithm
    Unsupervised Machine Learning with no underlying statistical model

    Algorithm:
        1. start with k random centroids (use points within the data)
        2. Assign all observations to their closest centroid
        3. recalcuate the groups means and call these your new centroids
        4. repeat 2,3 until no more changes in centroids

    Assumptions:
        - variance distribution of each attribute is spherical
        - all variables have same variance
        - probablity of k clusters is the same (same distribution)

    Strengths:
        - converges quickly (for clustering...)
        - Space Complexity: nk

    Weaknesses:
        - all clusters are spheres with the same radius
        - can only find local solution
        - will find exactly k clusters even if none, more, or less exist
    '''

    def __init__(self, k=4, max_iterations=258):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, data):
        '''
        runs the KMeans classification algorithm on a dataframe
        ---
        Inputs:
            @param data: pandas dataframe with n observations and p attributes
                         data should be scaled for best results
        ---
        Outputs:
            saves centroids and memberships to object.
            centroids: the center/mean of the different groups
            memberships: classification membership list for every observation
                         in the sample. This classification is only a local
                         solutation because centroids are assined randomly
        '''
        valid_data = self._check_input(data)
        if not valid_data:
            raise ValueError('Data cannot be empty or contain nulls')

        # Select a list of centroids to be the groups
        # implementation will use random observations for centroid
        # initialization
        centroids = array(data.sample(n=self.k, axis=0))
        membership = Series(index=data.index)

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
                dist = self._euclidian_distance(data.values, centroids[i])
                distance.loc[:,i] = dist

            # assign group memberships
            membership = distance.idxmin(axis=1)

            # check exit condition
            converged = self._check_convergence(data, centroids, membership)

            iterations += 1
            if iterations == self.max_iterations:
                break

        self.centroids = centroids
        self.membership = membership

    def fit_predict(self, data):
        '''
        runs the KMeans classification algorithm on a dataframe
        ---
        Inputs:
            @param data: pandas dataframe with n observations and p attributes
        ---
        Outputs:
            centroids: the center/mean of the different groups
            memberships: classification membership list for every observation
                         in the sample. This classification is only a local
                         solutation because centroids are assined randomly
        '''
        self.fit(data)
        return self.centroids, self.membership

    def _check_input(self, data):
        '''
        checks to see if the data can be processed
        ---
        Inputs:
            @param data: pandas dataframe with n observations and p attributes
        ---
        outputs:
            False if euclidian distance cannot be calculated, True otherwise
        '''
        if data.empty or (data.count() != len(data)).any():
            return False
        else:
            return True

    def _check_convergence(self, data, centroids, membership):
        '''
        check to see if the kmeans algorithm has converged
        convergence occurs when the 
        ---
        inputs:
            @param data: pandas dataframe with n observations and p attributes
        '''
        #calcuate new group centroids
        old_centroids = centroids.copy()
        
        # update clusters median
        for i in range(0,self.k):
            # update the centroid's range
            centroids[i] = data.where(membership == i).mean(axis=0).values

        # check for convergence
        if np_sum(centroids == old_centroids, axis=1).all():
            return True
        else:
            return False

    def _euclidian_distance(self, data, centroid):
        '''
        find the euclidean distance between data and centroid
        ---
        inputs:
            @param data: numpy 2D array of data observations
            @param centroid: numpy array representing one point
        ---
        output:
            list of distances from observations to centroid
        ---
        Assumptions:
            - data observations and centroid must be same dimension
            - data and centroid should not contain NaN
        Formula for Euclidean Distance:
            dist = sqrt(sum((a[1]+b[1])^2 + ... + (a[n]+b[n])^2))
        '''
        return sqrt(np_sum(power(data - centroid, 2), axis=1))
