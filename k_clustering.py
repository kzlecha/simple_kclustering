from pandas import DataFrame, Series
from numpy import array, power, sqrt
from numpy import sum as np_sum


def scale(data, method=None):
    '''
    scale the data for use in KMeans algorithm
    KMeans needs scaled data for optimal work
    ---
    input:
        @param data: pandas dataframe with n observations and p attributes
    ---
    output:
        modify input to be a scaled dataframe
    '''
    # check if desired scale is currently implemented
    if not method in [None, "MinMax", "Standardize"]:
        raise ValueError("Acceptable scale methods are:"+
                            "[False, MinMax, Standardize]")

    # scale
    if method == "MinMax":
        return (data-data.min())/(data.max() - data.min())
    if method == "Standardize":
        return (data-data.mean())/data.std()
    else:
        return data


class KClustering:
    '''
    Superclass for KMeans and KMedian Clustering algorithms
    '''
    def __init__(self, k=4, max_iterations=258):
        self.k = k
        self.max_iterations=max_iterations
    
    def _setup(self, data):
        '''
        set up necessary items for data processing.
        ---
        inputs:
            @param data: pandas dataframe with n observations and p attributes
        ---
        outputs:
            centroids: list of randomly initialized centroids
            membership: initalized Series denoting what group each obversation
                        belongs to. Note that it is initialized to NaN
        '''
        # check if data contains NaN
        valid_data = self._check_input(data)
        if not valid_data:
            raise ValueError('Data cannot be empty or contain nulls')

        # Select a list of centroids to be the groups
        # implementation will use random observations for centroid
        # initialization
        centroids = array(data.sample(n=self.k, axis=0))
        membership = Series(index=data.index, dtype="int")
        return centroids, membership

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

    def fit(self, data):
        pass
    
    def fit_predict(self, data):
        pass

    def predict(self, data):
        pass
