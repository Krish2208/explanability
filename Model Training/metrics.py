import numpy as np
import scipy
import sklearn
import sklearn.cluster
from sklearn.metrics import log_loss
import tqdm

def calc_identity(exp1, exp2):
    dis = np.array([np.array_equal(exp1[i],exp2[i]) for i in range(len(exp1))])
    total = dis.shape[0]
    true = np.sum(dis)
    score = (total-true)/total
    return score*100, true, total

def calc_separability(exp):
    wrong = 0
    for i in range(exp.shape[0]):
        for j in range(exp.shape[0]):
            if i == j:
                continue
            eq = np.array_equal(exp[i],exp[j])
            if eq:
                wrong = wrong + 1
    total = exp.shape[0]
    score = 100*abs(wrong)/total**2
    return wrong,total,total**2,score

def calc_stability(exp, labels):
    total = labels.shape[0]
    label_values = np.unique(labels)
    n_clusters = label_values.shape[0]
    init = np.array([[np.average(exp[np.where(labels == i)], axis = 0)] for i in label_values]).squeeze()
    ct = sklearn.cluster.KMeans(n_clusters = n_clusters, random_state=1, n_init=10, init = init)
    ct.fit(exp)
    error = np.sum(np.abs(labels-ct.labels_))
    if error/total > 0.5:
        error = total-error
    return error, total

def normalize_test(X_train, X_test):
    X_test_norm = X_test.copy()
    for i in X_train.columns:
        scaler = sklearn.preprocessing.MinMaxScaler()
        scaler.fit(X_train[i].values.reshape(-1,1))
        X_test_norm[i] = scaler.transform(X_test[i].values.reshape(-1,1))

    return X_test_norm

def calc_similarity(exp, X_test_norm):
    dbscan = sklearn.cluster.DBSCAN(eps=0.5, min_samples=10)
    dbscan.fit(X_test_norm[:400])
    labels = dbscan.labels_
    mean_dist = []
    for i in np.unique(labels):
        mean_dist.append(np.mean(sklearn.metrics.pairwise_distances(exp[np.where(labels == i), :, 1].squeeze(), metric='euclidean')))
    return np.min(mean_dist)

def permute(x, x_dash):
    x = x.copy()
    x_dash = x_dash.copy()
    x_rand = np.random.random(x.shape[0])
    x_new = [x[i] if x_rand[i] > 0.5 else x_dash[i] for i in range(len(x))]
    x_dash_new = [x_dash[i] if x_rand[i] > 0.5 else x[i] for i in range(len(x))]
    return x_new, x_dash_new

def calc_trust_score(model, test_x, exp, m, feat_list):
    total_recalls = []
    for i in tqdm.tqdm(range(len(test_x))):
        feat_score = np.zeros((len(feat_list)))
        for _ in range(m):
            x = test_x[i].copy()
            x_dash = test_x[np.random.randint(0,len(test_x))].copy()
            x_perm, x_dash_perm = permute(x, x_dash)
            for j in range(len(feat_list)):
                z = np.concatenate((x_perm[:j+1], x_dash_perm[j+1:]))
                z_dash = np.concatenate((x_dash_perm[:j], x_perm[j:]))
                p_z = model.predict_proba(z.reshape(1,-1))
                p_z_dash = model.predict_proba(z_dash.reshape(1,-1))
                feat_score[j] = feat_score[j] + np.linalg.norm(p_z-p_z_dash)
        feat_score = feat_score/m
        gold_feat_fs = np.argpartition(feat_score, -6)[-6:]
        recall = len(set(exp[i][:6, 0]).intersection(set(gold_feat_fs)))/6
        total_recalls.append(recall)
    return np.mean(total_recalls)

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