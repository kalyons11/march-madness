class Data:
    """Immutable data obejct storing all data about tournaments, for all years.
    """

    def __init__(self, teams, reg, tourney, seeds, slots, **kwargs):
        self.teams = teams
        self.reg = reg
        self.tourney = tourney
        self.seeds = seeds
        self.slots = slots
        self.kwargs = kwargs if kwargs is not None else {}