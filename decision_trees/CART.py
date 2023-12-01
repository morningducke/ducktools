import numpy as np
import graphviz
from abc import ABCMeta, abstractmethod

class ImpurityFunctions():
    """Class that contains impurity functions implementations"""
    def gini_index(y):
        """Calculate gini index for y

        Parameters:
            y: 1D np.array
        """
        if y.shape[0] == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        return 1 - np.sum((counts**2)) / (y.shape[0]**2)
    
    def variance(y):
        """Calculate variance for y

        Parameters:
            y: 1D np.array
        """
        return y.var() if y.shape[0] > 0 else 0

class DecisionNode:
    """Structure for a decision tree node
    
    Parameters:
        feature: Index of a feature on which the data is split on
        impurity: Impurity value of the split
        children: Each split as a node of this one 
        split_point: In case of a numerical feature - a float/integer value representing the split point in an inequality X >= split_point; in case of a categorical feature - list of unique values of that feature
        samples: Total amount of samples at this node
        pred: If this node is a leaf then this stores the predicted class or value
    """

    def __init__(self, feature=None, impurity=None, children=None, split_point=None, samples=None, pred=None):
        self.feature = feature
        self.impurity = impurity
        self.children = children
        self.split_point = split_point
        self.samples = samples
        self.pred = pred

    
    def graphviz_string(self, feature_dict, class_dict=None):
        """Converts the node to a graphviz node string

        Parameters:
            feature_dict: Feature to name mapping
            class_dict: Class to name mapping (None if regression)
        """
        # if a leaf
        if self.pred is not None:
            prediction = np.round(self.pred, 5) if class_dict is None else class_dict[self.pred]
            return f"impurity={np.round(self.impurity, 5)}\nsamples={self.samples}\npredicted: {prediction}\n"    
        
        return f"impurity={np.round(self.impurity, 5)}\nsamples={self.samples}\n{feature_dict[self.feature]} >= {self.split_point}\n"

class DecisionTreeBase(metaclass=ABCMeta):
    """Base for a decision tree classifier or regressor

    Parameters:
        max_depth: Stopping criteria, max depth of a resulting tree
        min_samples: Stopping criteria, the least amount of samples for a split to happen
        impurity_func: Function to calculate impurity of a node. Takes one parameter (y - targets) and returns a float representing impurity 
    """
    def __init__(self, max_depth=10, min_samples=2, impurity_func=None):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self._impurity_func = impurity_func
        self.root = None

    @abstractmethod
    def _calc_pred(self, y):
        """Calculates the prediction in case of a leaf node
        
        Parameters:
            y: Targets
        """
        pass

    def _split_on_feature(self, X, y, feature):
        """Returns the best split for a given feature, its split point and impurity

        Parameters:
            X: Data points
            y: Targets
            feature: Index of a feature on which to split on
        """
        split = []
        split_point = None
        impurity = None
        # if categorical feature split on each unique value of the feature
        if feature in self.cat_features:
            impurity = 0
            unique_vals = np.unique(X[:, feature])
            impurities = np.zeros(unique_vals.shape[0])
            split_point = unique_vals
            for idx, feature_val in enumerate(unique_vals):
                X_sub = X[X[:, feature] == feature_val]
                y_sub = y[X[:, feature] == feature_val]
                impurities[idx] += self._impurity_func(y_sub)
                impurity += impurities[idx] * y_sub.shape[0]
                split.append([X_sub, y_sub])
            impurity /= X.shape[0] # averaging impurity
        # if numerical feature split into two on a threshhold
        else:
            impurity = float("inf")
            threshholds = np.unique(X[:, feature])
            for threshhold in threshholds:
                y_sub_l = y[X[:, feature] >= threshhold]
                y_sub_r = y[X[:, feature] < threshhold]
                cur_impurity = self._impurity_func(y_sub_l) * y_sub_l.shape[0] + self._impurity_func(y_sub_r) * y_sub_r.shape[0]
                cur_impurity /= y.shape[0]
                if cur_impurity < impurity:
                    X_sub_l = X[X[:, feature] >= threshhold]
                    X_sub_r = X[X[:, feature] < threshhold]
                    split =[[X_sub_l, y_sub_l], [X_sub_r, y_sub_r]]
                    split_point = threshhold
                    impurity = cur_impurity
        return split, split_point, impurity
            
        
    def _build_tree(self, X, y, depth=0):
        """Builds a decision tree performing splits based on impurity"""
        
        n_samples, n_features = np.shape(X)
        node_impurity = self._impurity_func(y)
        if n_samples < self.min_samples or depth >= self.max_depth or node_impurity == 0:
            return DecisionNode(impurity=node_impurity, samples=n_samples, pred=self._calc_pred(y))

        best_feature = None
        best_split_point = None
        best_split = None
        best_impurity = float("inf")
        # trying a split on every feature 
        for feature_idx in range(n_features):
            split, split_point, impurity = self._split_on_feature(X, y, feature_idx)
            if impurity < best_impurity:
                best_feature = feature_idx
                best_split_point = split_point
                best_split = split
                best_impurity = impurity
                
        children = [self._build_tree(X, y, depth=depth+1) for X, y in best_split]
        return DecisionNode(feature=best_feature, impurity=node_impurity, children=children, split_point=best_split_point, samples=n_samples)

    def fit(self, X, y, cat_features=[]):
        """Fits (builds) the tree to data

        Parameters:
            X: training data (n_samples x n_features)
            y: targets
            cat_features (optional): indices of categorical features
        """
        self.cat_features = set(cat_features)
        self.root = self._build_tree(X, y)

    def predict_one(self, x, node=None):
        """Predicts output for one sample

        Parameters:
            x: data sample
            node: current node in the tree (pass root in it when calling)
        """
        if node is None:
            if self.root is None:
                raise ValueError("Tree has not been built yet")
            node = self.root 
        if node.pred is not None:
            return node.pred
            
        if node.feature in self.cat_features:
            return self.predict_one(x, node.children[node.split_point.index(x[node.feature])])
        else:
            if x[node.feature] >= node.split_point:
                return self.predict_one(x, node.children[0])
            else:
                return self.predict_one(x, node.children[1])
                
        
    def predict(self, X):
        """Predicts outputs for every sample in X

        Parameters:
            X: data (n_samples, n_features)

        Returns:
            list: list of predictions
        """
        return [self.predict_one(x, self.root) for x in X]
        
    def print_tree(self, feature_dict, class_dict, node=None, indent=""):
        """Recursively print the decision tree in standart output"""
        if not node:
            node = self.root

        # If we're at leaf => print the label
        if node.pred is not None:
            print(f"{indent}impurity={node.impurity}\n{indent}samples={node.samples}\n{indent}predicted: {class_dict[node.pred]}\n")
        # Go deeper down the tree
        else:
            print(f"{indent}impurity={node.impurity}\n{indent}samples={node.samples}\n{indent}split on {feature_dict[node.feature]} >= {node.split_point}\n")
            for child in node.children:
                self.print_tree(feature_dict, class_dict, child, indent+"--")

    def graphviz_tree(self, feature_dict, class_dict=None):
        """Convert the tree to a graphviz graph"""
        if self.root is None:
            raise ValueError("Tree has not been built yet")
        g = graphviz.Graph("Decision Tree")
        g.node('0', self.root.graphviz_string(feature_dict, class_dict), shape='rectangle', style='rounded')
        id = 1
        stack = [(self.root, 0)] # stack of node, parent id
        while len(stack) > 0:
            cur, p_id = stack.pop()
            if cur.children is None:
                continue
            for child in cur.children:
                g.node(str(id), child.graphviz_string(feature_dict, class_dict), shape='rectangle', style='rounded')
                g.edge(str(p_id), str(id))
                stack.append((child, id))
                id += 1
        return g
        
        
class ClassifierTree(DecisionTreeBase):
    """A classifier tree that uses gini impurity to build the tree and majority vote to get predictions on leaves

    Example:
    '''
    from CART import ClassifierTree
    from sklearn.datasets import load_iris
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    
    data = load_iris()
    X = data['data']
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    tc = ClassifierTree()
    tc.fit(X_train, y_train)

    # tree image output
    feature_dict = dict(enumerate(data.feature_names))
    class_dict = dict(enumerate(data.target_names))
    g = tc.graphviz_tree(feature_dict, class_dict)
    g

    preds = tc.predict(X_train)
    print(accuracy_score(y_train, preds))
    ConfusionMatrixDisplay(confusion_matrix(y_train, preds)).plot()
    '''
    """
    def __init__(self, max_depth=10, min_samples=2, impurity_func=None):
        super().__init__(max_depth=max_depth, min_samples=min_samples, impurity_func=ImpurityFunctions.gini_index)
        
    def _calc_pred(self, y):
        """Majority vote"""
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)] if counts.shape[0] > 0 else 0

class RegressionTree(DecisionTreeBase):
    """A classifier tree that uses mse impurity to build the tree and majority vote to get predictions on leaves"""
    def __init__(self, max_depth=10, min_samples=2, impurity_func=None):
        super().__init__(max_depth=max_depth, min_samples=min_samples, impurity_func=ImpurityFunctions.variance)

    def _calc_pred(self, y):
        """Mean over all targets"""
        return y.mean() if y.shape[0] > 0 else 0
        