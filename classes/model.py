import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .feature import Feature
from .feature_vector import FeatureVector


class Model:
    """Represents one of our models."""

    def __init__(self, name, features):
        """
        name: name of this model, for debugging purposes
        features: list[Feature] for this model
        """
        self.name = name
        self.features = features
        self._sklearn_model = None

    @property
    def sklearn_model(self):
        if self._sklearn_model is None:
            self._sklearn_model = LogisticRegression()
        return self._sklearn_model

    @classmethod
    def default(cls):
        return cls('default_linear',
                   [Feature('ppg', lambda df: df['score'].mean())])

    def get_vector(self, season, team, dm):
        """
        season: Season object
        team: Team object
        dm: DataManager
        returns: FeatureVector filtered on season and team, aggregated
            accordingly
        """
        filt = dm.get_team_in_season(season, team)
        # compute each feature and append to result dict
        result = {}
        for f in self.features:
            result[f] = f.compute(filt)
        return FeatureVector(result)

    def get_X_y(self, trainer):
        X, y = [], []
        # X is list of FeatureVector
        # y is 1/0 values
        data_raw = trainer.dm.get_training_data()
        for result in data_raw:
            vect_a = self.get_vector(result.season, result.winner, trainer.dm)
            vect_b = self.get_vector(result.season, result.loser, trainer.dm)

            vect_combo_a = self.combine_vectors(vect_a, vect_b)
            X.append(vect_combo_a.to_list())
            y.append(1)

            vect_combo_b = self.combine_vectors(vect_b, vect_a)
            X.append(vect_combo_b.to_list())
            y.append(0)
        return np.array(X), np.array(y)

    def train(self, trainer):
        # Need to do the following:
        # 1. Parse data
        X, y = self.get_X_y(trainer)
        # 2. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        # 3. Fit on training
        self.sklearn_model.fit(X_train, y_train)
        # 3. Evaluate on testing
        # 4. Return evaluation result
        return self.sklearn_model.score(X_test, y_test)

    def predict(self, a, b, runner):
        """
        a: Team a
        b: Team b
        runner: runner calling me
        returns: output in [0, 1] range, P(a)
        """
        vect_a = self.get_vector(runner.season, a, runner.dm)
        vect_b = self.get_vector(runner.season, b, runner.dm)
        vect_combo = self.combine_vectors(vect_a, vect_b)

    def combine_vectors(self, a, b):
        """
        a, b: FeatureVector's
        returns: combination of a and b
        """
        return a - b
