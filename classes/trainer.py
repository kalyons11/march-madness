from .data_manager import DataManager


class Trainer:
    """Trains a model."""

    def __init__(self, model, dm=None):
        self.model = model
        self.dm = dm if dm is not None else DataManager()

    def run(self):
        """trains and returns eval result"""
        return self.model.train(self)
