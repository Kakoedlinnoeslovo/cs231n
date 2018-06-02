import numpy as np


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def euclidean_two_loops(self, one, two):
        result = np.sqrt(np.sum((one - two) ** 2))
        return result

    def l1_two_loops(self, one, two):
        result = np.sum(np.abs(one - two))
        return result

    def euclidean_one_loop(self, one):
        # num_objects, num_fetures
        result_vector = np.sqrt(np.sum((self.X_train - one) ** 2, axis=1))
        return result_vector

    def euclidean_no_loops(self, X):
        x = self.X_train
        y = X
        m = x.shape[0]  # x has shape (m, d)
        n = y.shape[0]  # y has shape (n, d)
        x2 = np.sum(x ** 2, axis=1).reshape((m, 1))
        y2 = np.sum(y ** 2, axis=1).reshape((1, n))
        xy = x.dot(y.T)  # shape is (m, n)
        dists = np.sqrt(x2 + y2 - 2 * xy)  # shape is (m, n)
        return dists

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #dists_ = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension.                                    #
                #####################################################################
                # pass
                #####################################################################
                #                       END OF YOUR CODE                            #
                #####################################################################
                dists[i, j] = self.euclidean_two_loops(self.X_train[j], X[i])
                #dists_[i,j] = self.l1_two_loops(self.X_train[j], X[i])
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            #######################################################################
            # pass
            #######################################################################
            #                         END OF YOUR CODE                            #
            #######################################################################'
            dists[i, :] = self.euclidean_one_loop(X[i])
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # pass
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        dists = np.swapaxes(self.euclidean_no_loops(X), 0, 1)
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        sorted_dists = np.argsort(dists, axis=1)

        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            pass
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # pass
            #########################################################################
            #                           END OF YOUR CODE                            #
            #########################################################################
            y_pred[i] = np.median([self.y_train[p] for p in sorted_dists[i, :k]])

        return y_pred

if __name__ == "__main__":
    import numpy as np

    X_train = np.array([[1,2,3], [5,6,7]])
    y_train = np.array([0,1])

    X_train = (X_train - np.mean(X_train))

    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)

    X_test = np.array([[1,2,3], [5,6,7]])
    # Test your implementation:
    X_test = (X_test - np.mean(X_test)) / X_test.std()
    #dists, dists_ = classifier.compute_distances_two_loops(X_test)

    #assert (dists == dists_).all()
    #y_test_pred = classifier.predict_labels(dists, k=1)


