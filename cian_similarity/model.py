from pprint import pprint

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from cian_similarity.utils import calc_metrics, get_connection, get_features, get_offers, get_pairs


class Model:
    RANDOM_STATE_SKLEARN = 42
    TARGET = "resolution"

    def __init__(self, model_path=None):
        self._conn = None
        self.clf = LGBMClassifier()

        if model_path is not None:
            self.load(model_path)

    def _init(self):
        left_right = set((self.pairs.offer_id1 + self.pairs.offer_id2).values)
        right_left = set((self.pairs.offer_id2 + self.pairs.offer_id1).values)
        print("All pairs are unique:\t", (len(left_right.union(right_left)) / 2) == len(left_right))
        print("Mean prediction:\t", self.pairs.resolution.mean())
        print("Pairs shape:\t\t", self.pairs.shape)

    def train(self) -> None:
        X = self.pairs.apply(self.get_residual, axis=1)
        X = X.join(self.pairs[[self.TARGET]])

        X_train, X_test = train_test_split(X, stratify=X.resolution, random_state=self.RANDOM_STATE_SKLEARN)
        # train / test(==val)
        self.y_train = X_train[self.TARGET]
        self.y_test = X_test[self.TARGET]

        X_train = X_train.loc[:, X_train.columns != self.TARGET]
        X_test = X_test.loc[:, X_test.columns != self.TARGET]

        self.X_train = X_train
        self.X_test = X_test

        # model
        self.clf.fit(self.X_train, self.y_train)

        preds = self.clf.predict(X_test)
        pprint(calc_metrics(preds, self.y_test))

    def get_residual(self, row: pd.Series) -> pd.Series:
        left = self.feats.loc[row.offer_id1]
        right = self.feats.loc[row.offer_id2]

        residual = abs(left - right)
        residual = residual.fillna(-1)
        residual["totalarea_diff"] = residual["totalarea"] / max(left["totalarea"], right["totalarea"])

        return residual

    def get_residual_inference(self, left: pd.Series, right: pd.Series) -> pd.Series:
        residual = abs(left - right)
        residual = residual.fillna(-1)
        residual["totalarea_diff"] = residual["totalarea"] / max(left["totalarea"], right["totalarea"])

        return residual

    def predict(self, features: pd.Series) -> str:
        return self.clf.predict_proba()

    def save(self, path="model.pkl"):
        joblib.dump(self.clf, path)

    def load(self, path="model.pkl"):
        self.clf = joblib.load(path)

    @property
    def conn(self):
        if self._conn is None:
            self._conn = get_connection()

        return self._conn

    @property
    def feature_imporances(self):
        return pd.Series(self.clf.feature_importances_, index=self.X_train.columns)

    @property
    def offers(self):
        if not hasattr(self, "_offers"):
            self._offers = get_offers(self.conn)

        return self._offers

    @property
    def pairs(self):
        if not hasattr(self, "_pairs"):
            self._pairs = get_pairs(self.conn)

        return self._pairs

    @property
    def feats(self):
        if not hasattr(self, "_feats"):
            self._feats = get_features(self.offers)

        return self._feats
