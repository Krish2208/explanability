import numpy as np
import sklearn.metrics
import sklearn.cluster

def calc_identity_rules(top_rules1, top_rules2):
    dis = np.array([np.array_equal(top_rules1[i],top_rules2[i]) for i in range(len(top_rules1))])
    total = dis.shape[0]
    true = np.sum(dis)
    score = (total-true)/total
    return score*100, true, total

def calc_separability_rules(top_rules):
    wrong = 0
    for i in range(top_rules.shape[0]):
        for j in range(top_rules.shape[0]):
            if i == j:
                continue
            eq = np.array_equal(top_rules[i],top_rules[j])
            if eq:
                wrong = wrong + 1
    total = top_rules.shape[0]
    score = 100*abs(wrong)/total**2
    return wrong,total,total**2,score

def exp_enc(clf, exp):
    enc_exp = np.zeros((len(exp),len(clf.feature_names_)))
    for i in range(len(exp)):
        try:
            for j in range(len(clf.feature_names_)):
                if clf.feature_names_[j] in clf.rules_without_feature_names_[int(exp[i])][0].split():
                    enc_exp[i][j] = 1                    
        except:
            pass
    return enc_exp

def calc_stability_rules(top_rules, labels):
    total = labels.shape[0]
    label_values = np.unique(labels)
    n_clusters = label_values.shape[0]
    init = np.array([[np.average(top_rules[np.where(labels == i)], axis = 0)] for i in label_values]).squeeze()
    ct = sklearn.cluster.KMeans(n_clusters = n_clusters, random_state=1, n_init=10, init = init)
    ct.fit(top_rules)
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
    dbscan.fit(X_test_norm)
    labels = dbscan.labels_
    mean_dist = []
    for i in np.unique(labels):
        mean_dist.append(np.mean(sklearn.metrics.pairwise_distances(exp[np.where(labels == i), :].squeeze(), metric='euclidean')))
    return np.min(mean_dist)