import numpy as np


class DecisionTree:
    '''Decision Tree Classifier.

    Note that this class only supports binary classification.
    '''

    def __init__(self,
                 criterion,
                 max_depth,
                 min_samples_leaf,
                 sample_feature=False):
        '''Initialize the classifier.

        Args:
            criterion (str): the criterion used to select features and split nodes.
            max_depth (int): the max depth for the decision tree. This parameter is
                a trade-off between underfitting and overfitting.
            min_samples_leaf (int): the minimal samples in a leaf. This parameter is a trade-off
                between underfitting and overfitting.
            sample_feature (bool): whether to sample features for each splitting. Note that for random forest,
                we would randomly select a subset of features for learning. Here we select sqrt(p) features.
                For single decision tree, we do not sample features.
        '''
        if criterion == 'infogain_ratio':
            self.criterion = self._information_gain_ratio
        elif criterion == 'entropy':
            self.criterion = self._information_gain
        elif criterion == 'gini':
            self.criterion = self._gini_purification
        else:
            raise Exception('Criterion should be infogain_ratio or entropy or gini')
        self._tree = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sample_feature = sample_feature

    def fit(self, X, y, sample_weights=None):
        """Build the decision tree according to the training data.

        Args:
            X: (pd.Dataframe) training features, of shape (N, D). Each X[i] is a training sample.
            y: (pd.Series) vector of training labels, of shape (N,). y[i] is the label for X[i], and each y[i] is
            an integer in the range 0 <= y[i] <= C. Here C = 1.
            sample_weights: weights for each samples, of shape (N,).
        """
        if sample_weights is None:
            # if the sample weights is not provided, then by default all
            # the samples have unit weights.
            sample_weights = np.ones(X.shape[0]) / X.shape[0]
        else:
            sample_weights = np.array(sample_weights) / np.sum(sample_weights)

        feature_names = X.columns.tolist()
        X = np.array(X)
        y = np.array(y)
        self._tree = self._build_tree(X, y, feature_names, depth=1, sample_weights=sample_weights)
        return self

    @staticmethod
    def entropy(y, sample_weights):
        """Calculate the entropy for label.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the entropy for y.
        """
        entropy = 0.0
        # begin answer
        N = y.shape[0]
        # p_k        
        '''
        Sample_weights version 
        '''
        p0 = sum((y==0) * sample_weights) / N 
        p1 = sum((y==1) * sample_weights) / N
        if p0 == 0:
            log_p0 = 0
        else:
            log_p0 = np.log2(p0)
        if p1 == 0:
            log_p1 = 0
        else: 
            log_p1 = np.log2(p1)

        entropy = -(p0 * log_p0 + p1 * log_p1)
        '''
        No sample_weights version 
        '''
        # p0 = sum(y==0) / N
        # p1 = sum(y==1) / N
        # if p0 == 0:
        #     log_p0 = 0
        # else:
        #     log_p0 = np.log2(p0)
        # if p1 == 0:
        #     log_p1 = 0
        # else: 
        #     log_p1 = np.log2(p1)
        # entropy = -(p0 * log_p0 + p1 * log_p1)
        return entropy

    def _information_gain(self, X, y, index, sample_weights):
        """Calculate the information gain given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the information gain calculated.
        """
        info_gain = 0
        # YOUR CODE HERE
        # begin answer
        N, D = X.shape
        info_gain = self.entropy(y, sample_weights)
        feature = X[:, index]  # get the feature 
        f_min, f_max = min(feature), max(feature)
        # print(info_gain)
        for v in range(f_min, f_max+1):
            if sum(feature==v) != 0:  # feature data available 
                sub_X, sub_y, sub_sample_weights = self._split_dataset(X, y, index, v, sample_weights)
                info_gain -= sum(feature==v) / N * self.entropy(sub_y, sub_sample_weights)
        # end answer
        return info_gain

    def _information_gain_ratio(self, X, y, index, sample_weights):
        """Calculate the information gain ratio given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the information gain ratio calculated.
        """
        info_gain_ratio = 0
        split_information = 0.0
        # YOUR CODE HERE
        # begin answer
        N, D = X.shape
        feature = X[:, index]
        f_min, f_max = min(feature), max(feature)
        iv = 0

        for v in range(f_min, f_max+1):
            sub_X, sub_y, sub_sample_weights = self._split_dataset(X, y, index, v, sample_weights)
            ratio = sum(sub_sample_weights) / sum(sample_weights)
            iv -= ratio * np.log2(ratio)
        info_gain_ratio = self._information_gain(X, y, index, sample_weights) / iv
        # end answer
        return info_gain_ratio

    @staticmethod
    def gini_impurity(y, sample_weights):
        """Calculate the gini impurity for labels.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the gini impurity for y.
        """
        gini = 1
        # YOUR CODE HERE
        # begin answer
        p0 = sum((y==0) * sample_weights) / N 
        p1 = sum((y==1) * sample_weights) / N
        gini = p0 * (1 - p0) + p1 * (1 - p1)
        # end answer
        return gini

    def _gini_purification(self, X, y, index, sample_weights):
        """Calculate the resulted gini impurity given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the resulted gini impurity after splitting by this feature.
        """
        new_impurity = 1
        # YOUR CODE HERE
        # begin answer
        new_impurity = self.gini_impurity(y, sample_weights)
        value = np.unique(X[:, index])
        for v in value:
            sub_X, sub_y, sub_sample_weights = self._split_dataset(X, y, index, v, sample_weights)
            property = sum(sub_sample_weights) / sum(sample_weights)
            new_impurity -= property * self.gini_impurity(sub_y, sub_sample_weights)
        # end answer
        return new_impurity

    def _split_dataset(self, X, y, index, value, sample_weights):
        """Return the split of data whose index-th feature equals value.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for splitting.
            value: the value of the index-th feature for splitting.
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (np.array): the subset of X whose index-th feature equals value.
            (np.array): the subset of y whose index-th feature equals value.
            (np.array): the subset of sample weights whose index-th feature equals value.
        """
        N, D = X.shape
        sub_X, sub_y, sub_sample_weights = X, y, sample_weights
        # YOUR CODE HERE
        # Hint: Do not forget to remove the index-th feature from X.
        # begin answer
        idx = X[:, index]==value
        sub_X = X[:, [i for i in range(D) if i != index]]
        sub_X = sub_X[idx, :]
        sub_y = y[idx]
        sub_sample_weights = sample_weights[idx]
        # end answer
        return sub_X, sub_y, sub_sample_weights

    def _choose_best_feature(self, X, y, sample_weights):
        """Choose the best feature to split according to criterion.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (int): the index for the best feature
        """
        best_feature_idx = 0
        # YOUR CODE HERE
        # Note that you need to implement the sampling feature part here for random forest!
        # Hint: You may find `np.random.choice` is useful for sampling.
        # begin answer
        N, D = X.shape
        best_feature_idx = -1 
        max_gain = -np.inf

        if self.sample_feature:
            num = int(np.round(np.sqrt(D)))
            features = np.random.choice(D, num, replace=False)
            X = X[:, features]

        N, D = X.shape
        for i in range(D):
            gain = self._information_gain(X, y, i, sample_weights)
            if gain > max_gain:
                max_gain = gain 
                best_feature_idx = i
        # end answer
        return best_feature_idx

    @staticmethod
    def majority_vote(y, sample_weights=None):
        """Return the label which appears the most in y.
        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).
        Returns:
            (int): the majority label
        """
        if sample_weights is None:
            sample_weights = np.ones(y.shape[0]) / y.shape[0]
        majority_label = y[0]
        # YOUR CODE HERE
        # begin answer
        dic = {}
        for i in range(y.shape[0]):
            if y[i] not in dic.keys():
                dic[y[i]] = 1
            else:
                dic[y[i]] += 1
        
        majority_label = max(dic, key = dic.get)
        # end answer
        return majority_label

    def _build_tree(self, X, y, feature_names, depth, sample_weights):
        """Build the decision tree according to the data.

        Args:
            X: (np.array) training features, of shape (N, D).
            y: (np.array) vector of training labels, of shape (N,).
            feature_names (list): record the name of features in X in the original dataset.
            depth (int): current depth for this node.
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (dict): a dict denoting the decision tree. 
            Example:
                The first best feature name is 'title', and it has 5 different values: 0,1,2,3,4. For 'title' == 4, the next best feature name is 'pclass', we continue split the remain data. If it comes to the leaf, we use the majority_label by calling majority_vote.
                mytree = {
                    'title': {
                        0: subtree0,
                        1: subtree1,
                        2: subtree2,
                        3: subtree3,
                        4: {
                            'pclass': {
                                1: majority_vote([1, 1, 1, 1]) # which is 1, majority_label
                                2: majority_vote([1, 0, 1, 1]) # which is 1
                                3: majority_vote([0, 0, 0]) # which is 0
                            }
                        }
                    }
                }
        """
        mytree = dict()
        # YOUR CODE HERE
        # TODO: Use `_choose_best_feature` to find the best feature to split the X. Then use `_choose_best_feature` to
        # get subtrees.
        # Hint: You may find `np.unique` is useful. build_tree is recursive.
        # begin answer
        N, D = X.shape
        if depth >= self.max_depth or X.shape[0] <= self.min_samples_leaf or len(feature_names) == 0 or np.unique(y).shape[0] == 1:
            return self.majority_vote(y, sample_weights)
        
        best_feature_idx = self._choose_best_feature(X, y, sample_weights)
        # print(X.shape)
        # print(feature_names)
        # print(best_feature_idx)
        # print(depth)
        best_feature = feature_names[best_feature_idx]
        # mytree = {feature_names[best_feature_idx]:{}}
        mytree[feature_names[best_feature_idx]] = {}
        value = np.unique(X[:, best_feature_idx])
        feature_names = [f for f in feature_names if f != best_feature]
        for v in value:
            sub_X, sub_y, sub_sample_weights = self._split_dataset(X, y, best_feature_idx, v, sample_weights)
            mytree[best_feature][v] = self._build_tree(sub_X, sub_y, feature_names, depth+1, sub_sample_weights)

        # end answer
        return mytree

    def predict(self, X):
        """Predict classification results for X.

        Args:
            X: (pd.Dataframe) testing sample features, of shape (N, D).

        Returns:
            (np.array): predicted testing sample labels, of shape (N,).
        """
        if self._tree is None:
            raise RuntimeError("Estimator not fitted, call `fit` first")

        def _classify(tree  , x):
            """Classify a single sample with the fitted decision tree.

            Args:
                x: ((pd.Dataframe) a single sample features, of shape (D,).

            Returns:
                (int): predicted testing sample label.
            """
            # YOUR CODE HERE
            # begin answer
            parents = list(tree.keys())[0]
            child = tree[parents]
            key = x.loc[parents]
            if key not in child:
                key = np.random.choice(list(child.keys()))
            value = child[key]
            if isinstance(value, dict):
                label = _classify(value, x)
            else:
                label = value
            return label 
            # end answer

        # YOUR CODE HERE
        # begin answer
        output = []
        for x in range(X.shape[0]):
            output.append(_classify(self._tree, X.iloc[x, :]))
        return np.array(output)

        # end answer

    def show(self):
        """Plot the tree using matplotlib
        """
        if self._tree is None:
            raise RuntimeError("Estimator not fitted, call `fit` first")

        import tree_plotter
        tree_plotter.createPlot(self._tree)
