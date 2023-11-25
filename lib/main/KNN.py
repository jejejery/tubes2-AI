import numpy as np

def euclidean_distance(x1, x2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(x1 - x2)

class KNN:
    def __init__(self, k=3):
        """Initialize the KNN classifier with the number of neighbors (k)."""
        self.k = k

    def fit(self, X_train, y_train):
        """
        Train the KNN model with training data.

        Parameters:
        - X_train: Training feature vectors.
        - y_train: Training labels.
        """
        self.X_train = np.array(X_train)  # Convert training features to NumPy array
        self.y_train = np.array(y_train)  # Convert training labels to NumPy array

    def predict(self, X_test):
        """
        Predict labels for the given test data.

        Parameters:
        - X_test: Test feature vectors.

        Returns:
        - y_pred: Predicted labels for the test data.
        """
        X_test = np.array(X_test)
        if X_test.ndim == 1:
            # If a single data point is provided, convert it to a 2D array
            X_test = X_test.reshape(1, -1)

        y_pred = [self._predict(x) for x in X_test]
        return np.array(y_pred)

    def _predict(self, x):
        """
        Predict label for a single data point.

        Parameters:
        - x: Test feature vector for a single data point.

        Returns:
        - most_common: Predicted label for the input data point.
        """
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_neighbors_indices = np.argsort(distances)[:self.k]
        k_neighbor_labels = [int(self.y_train[i]) for i in k_neighbors_indices]
        most_common = np.bincount(k_neighbor_labels).argmax()
        return most_common
