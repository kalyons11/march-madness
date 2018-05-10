import pandas as pd

from .config import Config
from .data_manager import DataManager


class Runner:
    """Main class that will be used to get predictions for a particular
    year."""

    def __init__(self, season, model, dm=None):
        """
        season: Season object we want to run on
        model: Model object being used to make the predictions for this run
        dm: DataManager
        """
        self.season = season
        self.model = model
        self.dm = dm if dm is not None else DataManager()

    def run(self):
        """Generates all predictions for this season"""
        # Create result
        result = pd.DataFrame(columns=Config.PREDICTION_COLUMNS)
        # Get all teams in this season
        teams = self.dm.get_teams_in_season(self.season)
        # Iterate over all pairs
        for a in teams:
            for b in teams:
                if a != b:
                    game_id = self.get_game_id(a, b)
                    prediction = self.run_pair(a, b)
                    result = result.append({
                        result.columns[0]: game_id,
                        result.columns[1]: prediction
                    }, ignore_index=True)
        return result

    def run_pair(self, a, b):
        """
        a: first team
        b: second team
        returns: 1 if a wins, 0 else
        """
        return self.model.predict(a, b, self)

    def get_game_id(self, a, b):
        return str(self.season.yr) + '_' + str(a.team_id) + '_' + \
            str(b.team_id)
