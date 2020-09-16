from k_clustering import scale
from kmeans import KMeans
from kmedians import KMedians

from numpy import array
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal

import unittest

class TestKClustering(unittest.TestCase):
    '''
    Tests both implementations of KMeans Clustering
    '''

    # Scaling
    def test_scale(self):
        '''
        test to ensure minmax scaling and standardizing produce correct results
        the input and the calculated versions have different datatypes
        '''
        data = DataFrame([[1,0,1,0],
                          [2,1,3,0],
                          [0,2,1,0],
                          [5,6,5,6],
                          [7,6,6,7],
                          [7,5,6,7]])

        # check standardize scale
        scaled_data = DataFrame([[-0.866703, -1.253925, -1.140532, -0.908341],
                                 [-0.541689, -0.877747, -0.285133, -0.908341],
                                 [-1.191716, -0.501570, -1.140532, -0.908341],
                                 [0.433351,  1.003140,  0.570266,  0.726672],
                                 [1.083378,  1.003140,  0.997965,  0.999175],
                                 [1.083378,  0.626962,  0.997965,  0.999175]])

        test_data = scale(data, method="Standardize")
        assert_frame_equal(scaled_data, test_data, check_dtype=False)

        # check minmax scale
        scaled_data = DataFrame([[0.142857, 0.000000, 0.0, 0.000000],
                                 [0.285714, 0.166667, 0.4, 0.000000],
                                 [0.000000, 0.333333, 0.0, 0.000000],
                                 [0.714286, 1.000000, 0.8, 0.857143],
                                 [1.000000, 1.000000, 1.0, 1.000000],
                                 [1.000000, 0.833333, 1.0, 1.000000]])

        test_data = scale(data, method="MinMax")
        assert_frame_equal(scaled_data, test_data, check_dtype=False)

        # check no scale
        assert_frame_equal(data, scale(data), check_dtype=False)

    # KCLUSTERING METHODS
    def test_euc_dist(self):
        '''
        test to ensure euclidian distance is calculated correctly
        Formula:
            dist = sqrt(sum((a[1]+b[1])^2 + ... + (a[n]+b[n])^2))
        '''
        data = DataFrame([[1,0], [0,0]])
        vector = array([0,0])
        distance = [1,0]

        dist = KMeans()._euclidian_distance(data.values,vector)
        for i in range(0,1):
            self.assertEqual(dist[i], distance[i])

    def test_check_input(self):
        '''
        test to ensure that the input returns false if is empty or has NaN
        '''
        data = DataFrame()
        self.assertFalse(KMeans()._check_input(data))

        data = DataFrame([[0,1,2],
                          [None, 2, 3],
                          [3,4,5]])
        self.assertFalse(KMeans()._check_input(data))

        data = data.dropna()
        self.assertTrue(KMeans()._check_input(data))


    # KMEANS
    def test_kmeans(self):
        '''
        test to see if the kmeans clustering will crash.
        Since KMeans is a random algorithm that only finds a local solution
        attempting to determine the "correctness" is impossible
        '''
        data = DataFrame([[1,0,1,0],
                          [2,1,3,0],
                          [0,2,1,0],
                          [5,6,5,6],
                          [7,6,6,7],
                          [7,5,6,7]])
        
        # kmeans must take scaled data
        # standardize with mean normalization
        data=(data-data.mean())/data.std()

        k = 2

        # run kmeans
        kmeans = KMeans(k)
        centroids, memberships = kmeans.fit_predict(data)

        # membership must be in range of 0 to k-1 inclusive
        for membership in memberships:
            if membership not in range(0,k):
                assert False

    def test_kmeans_has_converged(self):
        '''
        test to ensure convergence occurs only when the centroids stop changing
        '''
        data = DataFrame([[0,0,0], [4,4,4], [-4,-4,-4]])
        centroids = DataFrame([[1,0,0], [3,4,3], [-2,-3,-4]]).values
        memberships = Series([0,1,2], index=data.index)

        self.assertFalse(KMeans(k=3)._has_converged(data,centroids,memberships))

        centroids = DataFrame([[0,0,0], [4,4,4], [-4,-4,-4]]).values
        self.assertTrue(KMeans(k=3)._has_converged(data,centroids,memberships))

    # KMEDIANS
    def test_kmedians(self):
        '''
        test to see if the kmeans clustering will crash.
        Since KMeans is a random algorithm that only finds a local solution
        attempting to determine the "correctness" is impossible
        '''
        data = DataFrame([[1,0,1,0],
                          [2,1,3,0],
                          [0,2,1,0],
                          [5,6,5,6],
                          [7,6,6,7],
                          [7,5,6,7]])
        
        # kmeans must take scaled data
        # standardize with mean normalization
        data=(data-data.mean())/data.std()

        k = 2

        # run kmeans
        kmedians = KMedians(k)
        centroids, memberships = kmedians.fit_predict(data)

        # membership must be in range of 0 to k-1 inclusive
        for membership in memberships:
            if membership not in range(0,k):
                assert False

    def test_kmedians_has_converged(self):
        '''
        test to ensure convergence occurs only when the centroids stop changing
        '''
        data = DataFrame([[0,0,0], [4,4,4], [-4,-4,-4]])
        centroids = DataFrame([[1,0,0], [3,4,3], [-2,-3,-4]]).values
        memberships = Series([0,1,2], index=data.index)

        self.assertFalse(KMedians(k=3)._has_converged(data,centroids,memberships))

        centroids = DataFrame([[0,0,0], [4,4,4], [-4,-4,-4]]).values
        self.assertTrue(KMedians(k=3)._has_converged(data,centroids,memberships))


# run the unittests
if __name__ == "__main__":
    unittest.main()
