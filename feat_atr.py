import numpy as np
import scipy
from sklearn.metrics import log_loss

class FeatureAttribution:
    def __init__(self, model, inst, y, sorted_atr):
        self.model = model
        self.inst = inst
        self.y = y
        self.sorted_atr = sorted_atr
        self.losses = []
        self.atr_values = []

    def monotonicity(self):
        losses = []
        atr_values = []
        for i in range(len(self.sorted_atr)):
            atr = self.sorted_atr[i]
            new_inst = np.copy(self.inst)
            np.put(new_inst, i, -1)
            loss = log_loss(self.y, self.model.predict_proba(new_inst.reshape(1, -1))[0])
            losses.append(loss)
            atr_values.append(abs(atr))
        self.losses = losses
        self.atr_values = atr_values
        monotonicity = scipy.stats.spearmanr(losses, atr_values).correlation
        return monotonicity

    def non_sensitivity(self):
        loss_zeros = set([i for i in range(len(self.losses)) if self.losses[i] == 0])
        atr_zeros = set([i for i in range(len(self.atr_values)) if self.atr_values[i] == 0])
        non_sensitivity = len(loss_zeros.symmetric_difference(atr_zeros))
        return non_sensitivity
    
    def effective_complexity(self, sorted_feat, threshold):
        min_k = 0
        threshold = 0.1
        for i in range(len(sorted_feat)):
            new_inst = np.copy(self.inst)
            for j in range(i+1, len(sorted_feat)):
                np.put(new_inst, sorted_feat[j], -1)
            loss = log_loss(self.y, self.model.predict_proba(new_inst.reshape(1, -1))[0])
            if loss < threshold:
                min_k = i+1
        return min_k