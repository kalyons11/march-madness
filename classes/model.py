import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split

from .feature import Feature
from .feature_vector import FeatureVector
from .game_context import GameContext


class Model:
    """Represents one of our models."""

    def __init__(self, name, features, model_type='LogisticRegression',
                 mode='sub'):
        """
        name: name of this model, for debugging purposes
        features: list[Feature] for this model
        model_type: .
        """
        self.name = name
        self.features = features
        self.model_type = model_type
        self._sklearn_model = None
        self.mode = mode

    @property
    def sklearn_model(self):
        if self._sklearn_model is not None:
            return self._sklearn_model
        elif self.model_type == "LogisticRegression":
            self._sklearn_model = LogisticRegression()
        elif self.model_type == "RandomForestClassifier":
            self._sklearn_model = RandomForestClassifier()
        return self._sklearn_model

    @classmethod
    def default(cls):
        return cls('default_linear',
                   [Feature('ppg', lambda df: df['score'].mean())])

    def get_vector(self, season, team, other, dm):
        """
        season: Season object
        team: Team object
        dm: DataManager
        returns: FeatureVector filtered on season and team, aggregated
            accordingly
        """
        # compute each feature and append to result dict
        result = {}
        for f in self.features:
            result[f] = f.compute(dm, team, GameContext(team, other, season))
        return FeatureVector(result)

    def get_X_y(self, trainer):
        X, y = [], []
        # X is list of FeatureVector
        # y is 1/0 values
        data_raw = trainer.dm.get_training_data()
        for result in data_raw:
            vect_a = self.get_vector(result.season, result.winner,
                                     result.loser, trainer.dm)
            vect_b = self.get_vector(result.season, result.loser,
                                     result.winner, trainer.dm)

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
        #import pdb; pdb.set_trace()
        # 3. Fit on training
        self.sklearn_model.fit(X_train, y_train)
        # 3. Evaluate on testing
        y_pred = self.sklearn_model.predict(X_test)
        # 3A. ROC/AUC
        self.generate_roc_auc_curve(y_test, y_pred)
        # 4. Return evaluation result
        return classification_report(y_test, y_pred)
        # score = self.sklearn_model.score(X_test, y_test)

    def predict(self, a, b, runner):
        """
        a: Team a
        b: Team b
        runner: runner calling me
        returns: 0/1 - 1 if a wins, 0 else
        """
        vect_a = self.get_vector(runner.season, a, b, runner.dm)
        vect_b = self.get_vector(runner.season, b, a, runner.dm)
        vect_combo = self.combine_vectors(vect_a, vect_b)
        X = [vect_combo.to_list()]
        result = self._sklearn_model.predict(X)
        return result[0]

    def combine_vectors(self, a, b):
        """
        a, b: FeatureVector's
        returns: combination of a and b
        """
        if self.mode == 'sub':
            return a - b
        elif self.mode == 'concat':
            return a.concat(b)

    def generate_roc_auc_curve(self, y_true, y_pred):
        lw = 2
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic of model {self.name}')
        plt.legend(loc="lower right")
        plt.show()
