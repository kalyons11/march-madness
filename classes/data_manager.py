from os.path import join as path_join

import numpy as np
import pandas as pd
from .config import Config
from .data import Data
from .season import Season
from .team import Team
from .tourney_result import TourneyResult


class DataManager:
    """Immutable loader that can load data from directory and query on
    specific fields."""

    def __init__(self, data_dir=Config.DEFAULT_DATA_DIR):
        """data_dir: path to dir with all the csv's we need"""
        self.data_dir = data_dir
        self._data = None  # full dataframe

    @property
    def data(self):
        """get the full dataframe for this loader"""
        if self._data is None:
            self._data = self.load()
        return self._data

    def load(self):
        """loads data into local memory for later use"""
        teams = pd.read_csv(path_join(self.data_dir, Config.TEAMS_FNAME))
        reg = pd.read_csv(path_join(self.data_dir, Config.REG_FNAME))
        tourney = pd.read_csv(path_join(self.data_dir, Config.TOURNEY_FNAME))
        seeds = pd.read_csv(path_join(self.data_dir, Config.SEEDS_FNAME))
        slots = pd.read_csv(path_join(self.data_dir, Config.SLOTS_FNAME))
        return Data(teams, reg, tourney, seeds, slots, key='value')

    # QUERIES

    def get_teams_in_season(self, season):
        reg = self.data.reg
        reg_season = reg[reg.Season == season.yr]
        reg_season_teams_winners = reg_season.Wteam.unique()
        reg_season_teams_losers = reg_season.Lteam.unique()
        reg_season_teams_all = np.union1d(
            reg_season_teams_winners, reg_season_teams_losers)
        return map(lambda t: Team(t, self), reg_season_teams_all)

    def get_team_in_season(self, season, team):
        reg = self.data.reg
        reg_season = reg[reg.Season == season.yr]
        reg_season_team = reg_season[(reg.Wteam == team.team_id) | (
            reg.Lteam == team.team_id)]
        return self.team_win_lose_score_helper(reg_season_team, team)

    def get_training_data(self):
        """
        gets all data for a model to train on
        need to iterate over all tourney games we have and give TourneyResults
        """
        tourney = self.data.tourney
        result = []
        for _, row in tourney.iterrows():
            winner = Team(row.Wteam, self)
            loser = Team(row.Lteam, self)
            season = Season(row.Season)
            result.append(TourneyResult(
                winner=winner, loser=loser, season=season))
        return result

    # HELPERS

    def team_win_lose_score_helper(self, df, team):
        """
        df: filtered on team and year
        team: team object
        returns: updated df with removing 'W'/'L' from relevant stats
        """
        rename_cols = {
            'team', 'score', 'fgm', 'fga', 'fgm3', 'fga3', 'ftm', 'fta', 'or',
            'dr', 'ast', 'to', 'stl', 'blk', 'pf'
        }

        def get_rename_dict(mode):
            """
            mode: string 'W' or 'L'
            """
            result = {}
            for col in rename_cols:
                result[mode + col] = col
            # Extra cols
            result['Season'] = 'season'
            return result

        def internal_update_func(row):
            """
            row: row of table we want to update
            """
            if row.Wteam == team.team_id:
                # Must add W to everything in rename cols
                rename_dict = get_rename_dict('W')
            else:
                # Must add L to everything in rename cols
                rename_dict = get_rename_dict('L')

            # Select subset
            subset = row[rename_dict.keys()]

            return subset.rename(rename_dict)

        return df.apply(internal_update_func, axis=1)
