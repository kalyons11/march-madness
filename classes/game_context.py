"""Context of a game.
"""


class GameContext(object):
    """context of a game between 2 teams"""
    def __init__(self, me, other, season, **kwargs):
        self.me = me
        self.other = other
        self.season = season
        self.kwargs = kwargs
