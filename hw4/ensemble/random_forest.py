import copy
import numpy as np


class RandomForest:
    '''Random Forest Classifier.

    Note that this class only support binary classification.
    '''

    def __init__(self,
                 base_learner,
                 n_estimator,
                 seed=2020):
        '''Initialize the classifier.

        Args:
            base_learner: the base_learner should provide the .fit() and .predict() interface.
            n_estimator (int): The number of base learners in RandomForest.
            seed (int): random seed
        '''
        np.random.seed(seed)
        self.base_learner = base_learner
        self.n_estimator = n_estimator
        self._estimators = [copy.deepcopy(self.base_learner) for _ in range(self.n_estimator)]

    def _get_bootstrap_dataset(self, X, y):
        """Create a bootstrap dataset for X.

        Args:
            X: training features, of shape (N, D). Each X[i] is a training sample.
            y: vector of training labels, of shape (N,).

        Returns:
            X_bootstrap: a sampled dataset, of shape (N, D).
            y_bootstrap: the labels for sampled dataset.
        """
        # YOUR CODE HERE
        # TODO: re-sample N examples from X with replacement
        # begin answer
        idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
        return X.iloc[idx, :], y.iloc[idx]
        # end answer

    def fit(self, X, y):
        """Build the random forest according to the training data.

        Args:
            X: training features, of shape (N, D). Each X[i] is a training sample.
            y: vector of training labels, of shape (N,).
        """
        # YOUR CODE HERE
        # begin answer
        for i in range(self.n_estimator):
            bootstrap_X, bootstrap_y = self._get_bootstrap_dataset(X, y)
            self._estimators[i].fit(bootstrap_X, bootstrap_y)
        self._y = np.unique(y)
        # end answer
        return self


    def predict(self, X):
        """Predict classification results for X.

        Args:
            X: testing sample features, of shape (N, D).

        Returns:
            (np.array): predicted testing sample labels, of shape (N,).
        """
        N = X.shape[0]
        y_pred = np.zeros(N)
        # YOUR CODE HERE
        # begin answer
        pred = [estimator.predict(X) for estimator in self._estimators]
        p = []  # store the probabilities of each label 
        for v in self._y:
            probability = np.mean(np.array(pred) == v, axis=0)
            p.append(probability)
        y_pred = self._y[np.argmax(np.array(p).T, axis=1)]
        # end answer
        return y_pred

