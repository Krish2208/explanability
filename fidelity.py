import pandas as pd
import numpy as np

class Fidelity:
    
    def __init__(self, clf, explainer, X, feature_names):
        self.clf = clf
        self.explainer = explainer
        self.X = X
        self.depth = 0
        self.feature_names = feature_names
        self.n_nodes = self.clf.tree_.node_count
        self.is_leaves = np.zeros(shape=self.n_nodes, dtype=bool)
        self.calc_depth()

    def calc_depth(self):
        children_left = self.clf.tree_.children_left
        children_right = self.clf.tree_.children_right
        node_depth = np.zeros(shape=self.n_nodes, dtype=np.int64)
        stack = [(0, 0)]
        while len(stack) > 0:
            node_id, depth = stack.pop()
            node_depth[node_id] = depth
            is_split_node = children_left[node_id] != children_right[node_id]
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                self.is_leaves[node_id] = True
        max_depth = max(node_depth)
        self.depth = max_depth

    def phase_one_util(self, sample_id):
        feature = self.clf.tree_.feature
        node_indicator = self.clf.decision_path(self.X)
        true_features = []
        node_index = node_indicator.indices[
            node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
        ]
        for i in node_index:
            if not self.is_leaves[i]:
                true_features.append(feature[i])

        exp_fn = lambda i: self.explainer.explain_instance(self.X[i, :], self.clf.predict_proba)
        inst_expl = exp_fn(sample_id)
        # explanation_features = inst_expl.as_list(label=inst_expl.available_labels()[0])
        explanation_features = inst_expl.as_map()[inst_expl.available_labels()[0]]
        explanation_features = sorted(explanation_features, key=lambda x: x[1], reverse=True)
        if len(explanation_features) > self.depth:
            recall_explanation_features = [x[0] for x in explanation_features[:self.depth]]
        elif len(explanation_features) > (2*self.depth)/3:
            recall_explanation_features = [x[0] for x in explanation_features[:(2*self.depth)//3]]
        else:
            recall_explanation_features = [x[0] for x in explanation_features]
        explanation_features_weights = np.array([x[1] for x in explanation_features])
        top_quartile = np.percentile(explanation_features_weights, 75)
        precision_explanation_features = [x[0] for x in explanation_features if x[1] >= top_quartile]

        true_features_set = set(true_features)
        precision_explanation_features_set = set(precision_explanation_features)
        recall_explanation_features_set = set(recall_explanation_features)

        recall = len(true_features_set.intersection(recall_explanation_features_set)) / len(true_features_set)
        precision = len(true_features_set.intersection(precision_explanation_features_set)) / len(precision_explanation_features_set)
        return recall, precision

    def phase_one(self):
        recall = []
        precision = []
        for i in range(5):
            r, p = self.phase_one_util(i)
            recall.append(r)
            precision.append(p)
        return recall, precision
    
    def phase_two_util(self, sample_id):
        feature = self.clf.tree_.feature
        node_indicator = self.clf.decision_path(self.X)
        true_features = []
        node_index = node_indicator.indices[
            node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
        ]
        for i in node_index:
            if not self.is_leaves[i]:
                true_features.append(feature[i])

        exp_fn = lambda i: self.explainer.explain_instance(self.X[i, :], self.clf.predict_proba)
        inst_expl = exp_fn(sample_id)
        # explanation_features = inst_expl.as_list(label=inst_expl.available_labels()[0])
        explanation_features = inst_expl.as_map()[inst_expl.available_labels()[0]]
        explanation_features = sorted(explanation_features, key=lambda x: x[1], reverse=True)
        opt_f1 = 0
        opt_dec = 0
        for dec in range(90, 0, -10):
            explanation_features_weights = np.array([x[1] for x in explanation_features])
            top_quartile = np.percentile(explanation_features_weights, dec)
            sample_explanation_features = [x[0] for x in explanation_features if x[1] >= top_quartile]

            true_features_set = set(true_features)
            sample_explanation_features_set = set(sample_explanation_features)

            recall = len(true_features_set.intersection(sample_explanation_features_set)) / len(true_features_set)
            precision = len(true_features_set.intersection(sample_explanation_features_set)) / len(sample_explanation_features_set)
            try:
                f1 = 2 * (precision * recall) / (precision + recall)
                if f1 > opt_f1:
                    opt_f1 = f1
                    opt_dec = dec
            except ZeroDivisionError:
                continue
        return opt_f1, opt_dec