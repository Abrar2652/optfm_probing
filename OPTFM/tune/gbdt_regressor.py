import numpy as np
from numpy import ndarray
from sklearn.ensemble import GradientBoostingRegressor
import math


class GradientBoostingRegressor_():
    '''
    Class Description:
    GBDT class, which stores the trained GBDT.
    '''
    def __init__(self, model, random_state, n_estimators, learning_rate, max_depth, min_samples_split):
        '''
        Function Description:
        Initialize the GBDT.
        
        Parameters:
        - n_estimators: Number of decision trees.
        - learning_rate: Learning rate.
        - max_depth: Maximum depth of the decision trees.
        - min_samples_split: Minimum number of samples required to split a leaf node.
        - subsample: Subsample rate without replacement.
        '''
        self.prediction_value = None
        self.logit = []
        self.loss = []
        if model:
            self.model = model
        else:
            self.model = GradientBoostingRegressor(random_state=random_state, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_samples_split=min_samples_split, verbose=1)

    def fit(self, data: ndarray, 
            label: ndarray):
        '''
        Function Description:
        Train the GBDT based on the given decision variable neural encoding and optimal solution values.

        Parameters:
        - data: Neural encoding results of the decision variables.
        - label: Values of the decision variables in the optimal solution.

        Return: 
        The training results are stored in the class. There is no return value.
        '''
        self.model.fit(X=data, y=label)

    def predict(self, data: ndarray) -> ndarray:
        '''
        Function Description:
        Use the trained GBDT to predict the initial solution based on the given decision variable neural encoding, and return the predicted initial solution.

        Parameters:
        - data: Neural encoding results of the decision variables.

        Return: 
        The predicted initial solution.
        '''
        self.prediction_value = self.model.predict(data)
        for i in range(len(self.prediction_value)):
            self.logit.append(math.floor(self.prediction_value[i] + 0.5))

        # return the round value for each variable
        return self.logit

    
    def calc(self, data: ndarray) -> ndarray:
        '''
        Function Description:
        Use the trained GBDT to predict the initial solution based on the given decision variable neural encoding, and return the prediction loss.

        Parameters:
        - data: Neural encoding results of the decision variables.

        Return: 
        The prediction loss generated when predicting the initial solution for each decision variable.
        '''

        # TODO: Cannot calculate the prediction loss, so implement as usual.
        for i in range(len(self.prediction_value)):
            self.loss.append(abs(self.prediction_value[i] - self.logit[i]))
        
        return self.loss
    
    def get_partition_result(self, tree_index, neighborhood_count, data):

        tree = self.model.estimators_[tree_index, 0]
        node_dict = {}
        for i in range(len(data)):
            node_indicator = tree.decision_path([data[i]])
            node_index = node_indicator.indices
            for idx in node_index:
                if idx not in node_dict.keys():
                    node_dict[idx] = set()
                node_dict[idx].add(i)
    
        tree_ = tree.tree_
        neighborhood_node_list = []
        neighborhood_node_list = [0]  # 初始化为根节点
        while neighborhood_node_list:
            node_index = neighborhood_node_list.pop(0)
            left_index = None
            right_index = None
            # 如果是叶子节点，则无需继续遍历
            if tree_.children_left[node_index] != -1:
                # 左子节点
                left_index = tree_.children_left[node_index]
            if tree_.children_right[node_index] != -1:
                # 右子节点
                right_index = tree_.children_right[node_index]
            
            # 将子节点加入队列
            if left_index:
                neighborhood_node_list.append(left_index)
            if right_index:
                neighborhood_node_list.append(right_index)
            
            if len(neighborhood_node_list) >= neighborhood_count:
                break

        neighborhood_list = []
        for idx in neighborhood_node_list:
            if idx in node_dict.keys():
                neighborhood_list.append(list(node_dict[idx]))
        
        return neighborhood_list