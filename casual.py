import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

class CasualExplanations:
    def __init__(self, exp, X, X_scaled, Y, categorical, numerical, classes, enc, ohe, x_min, x_max, rng, k=10, **kwargs):
        self.exp = exp
        self.X = X
        self.X_scaled = X_scaled
        self.Y = Y
        self.categorical = categorical
        self.numerical = numerical
        self.classes = classes
        self.enc = enc
        self.ohe = ohe
        self.x_min = x_min
        self.x_max = x_max
        self.rng = rng
        self.k = k
        if "low_limit" in kwargs:
            self.low_limit = kwargs["low_limit"]
        else:
            self.low_limit = 0
        if "high_limit" in kwargs:
            self.high_limit = kwargs["high_limit"]
        else:
            self.high_limit = len(X)
        self.cfs = self.generate_counterfactuals()
        self.X_enc = self.enc.predict(self.X_scaled)
        self.class_mean = {}
        for c in classes:
            self.class_mean[c] = self.X_enc[self.Y.argmax(axis=1)==c].mean(axis=0)
    
    def generate_counterfactuals(self):
        return self.exp.generate_counterfactuals(self.X[self.low_limit:self.high_limit], total_CFs=self.k, desired_class="opposite")
    
    def average_sparsity(self, id):
        sum_sparsity = 0
        for i in range(self.k):
            sum_sparsity += (self.cfs.cf_examples_list[id].final_cfs_df.iloc[i, :-1] != self.X.iloc[id]).sum()
        return sum_sparsity / self.k
    
    def total_average_sparsity(self):
        sum_sparsity = 0
        for i in range(self.low_limit, self.high_limit):
            sum_sparsity += self.average_sparsity(i)
        return sum_sparsity / (self.high_limit - self.low_limit)
    
    def average_proximity(self, id):
        sum_proximity = 0
        for i in range(self.k):
            proximity = 0
            for col in self.categorical:
                if self.cfs.cf_examples_list[id].final_cfs_df.iloc[i, :-1][col]!=self.X.iloc[id][col]:
                    proximity+=1
            for col in self.numerical:
                proximity += abs(self.cfs.cf_examples_list[id].final_cfs_df.iloc[i, :-1][col] - self.X.iloc[id][col])
            sum_proximity+=proximity
        return sum_proximity / self.k
    
    def total_average_proximity(self):
        sum_proximity = 0
        for i in range(self.low_limit, self.high_limit):
            sum_proximity += self.average_proximity(i)
        return sum_proximity / (self.high_limit - self.low_limit)
    
    def average_diversity(self, id):
        sum_diversity = 0
        for i in range(self.k-1):
            diversity = 0
            for j in range(i+1, self.k):
                for col in self.categorical:
                    if self.cfs.cf_examples_list[id].final_cfs_df.iloc[i, :-1][col]!=self.cfs.cf_examples_list[id].final_cfs_df.iloc[j, :-1][col]:
                        diversity+=1
                for col in self.numerical:
                    diversity += abs(self.cfs.cf_examples_list[id].final_cfs_df.iloc[i, :-1][col] - self.cfs.cf_examples_list[id].final_cfs_df.iloc[j, :-1][col])
            sum_diversity += diversity
        return sum_diversity / (self.k**2)
    
    def total_average_diversity(self):
        sum_diversity = 0
        for i in range(self.low_limit, self.high_limit):
            sum_diversity += self.average_diversity(i)
        return sum_diversity / (self.high_limit - self.low_limit)

    def average_interpretability(self, id):
        x_cf = self.cfs.cf_examples_list[id].final_cfs_df.iloc[:, :-1]
        y_cf = self.cfs.cf_examples_list[id].final_cfs_df.iloc[:, -1]
        x_cf = np.c_[self.ohe.transform(x_cf.loc[:, self.categorical]), (x_cf.loc[:, self.numerical] - self.x_min) / (self.x_max - self.x_min) * (self.rng[1] - self.rng[0]) + self.rng[0]].astype(np.float32, copy=False)
        cf_enc = self.enc.predict(x_cf)
        sum_interpretability = 0
        for i in range(self.k):
            dist_orig = np.linalg.norm(cf_enc[i] - self.class_mean[self.Y[id].argmax(axis=0)])
            dist_cf = np.linalg.norm(cf_enc[i] - self.class_mean[y_cf[i]])
            sum_interpretability += dist_orig / (dist_cf + 1e-10)
        return sum_interpretability / self.k

    def total_average_interpretability(self):
        sum_interpretability = 0
        for i in range(self.low_limit, self.high_limit):
            sum_interpretability += self.average_interpretability(i)
        return sum_interpretability / (self.high_limit - self.low_limit)


def casual_interpret_full(train: tuple, test: tuple, exp: dice_ml.explainer_interfaces.dice_random.DiceRandom, categorical : list, numerical : list, classes : list, k : int, **kwargs):
    x_train, y_train = train
    x_test, y_test = test
    x_train_num = x_train.loc[:, numerical].astype(np.float32, copy=False)
    x_train_cat = x_train.loc[:, categorical].copy()
    x_min, x_max = x_train_num.min(axis=0), x_train_num.max(axis=0)
    rng = (-1., 1.)
    ohe = OneHotEncoder(categories='auto', sparse=False).fit(x_train_cat)
    X_train = np.c_[ohe.transform(x_train_cat), (x_train_num - x_min) / (x_max - x_min) * (rng[1] - rng[0]) + rng[0]].astype(np.float32, copy=False)
    X_test = np.c_[ohe.transform(x_test.loc[:, categorical]), (x_test.loc[:, numerical] - x_min) / (x_max - x_min) * (rng[1] - rng[0]) + rng[0]].astype(np.float32, copy=False)
    Y_train = to_categorical(y_train)
    Y_test = to_categorical(y_test)
    def ae_model():
        # encoder``
        x_in = Input(shape=(29,))
        x = Dense(60, activation='relu')(x_in)
        x = Dense(30, activation='relu')(x)
        x = Dense(15, activation='relu')(x)
        encoded = Dense(10, activation=None)(x)
        encoder = Model(x_in, encoded)
        # decoder
        dec_in = Input(shape=(10,))
        x = Dense(15, activation='relu')(dec_in)
        x = Dense(30, activation='relu')(x)
        x = Dense(60, activation='relu')(x)
        decoded = Dense(29, activation=None)(x)
        decoder = Model(dec_in, decoded)
        # autoencoder = encoder + decoder
        x_out = decoder(encoder(x_in))
        autoencoder = Model(x_in, x_out)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder, encoder, decoder
    ae, enc, dec = ae_model()
    ae.fit(X_train, X_train, batch_size=128, epochs=10, validation_data=(X_test, X_test), verbose=1)
    train_ce = CasualExplanations(exp, x_train, X_train, Y_train, categorical, numerical, classes, enc, ohe, x_min, x_max, rng, k, **kwargs)
    test_ce = CasualExplanations(exp, x_test, X_test, Y_test, categorical, numerical, classes, enc, ohe, x_min, x_max, rng, k, **kwargs)
    return train_ce, test_ce