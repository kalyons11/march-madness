class Team:
    """Immutable notion of a team."""

    def __init__(self, team_id, dm):
        """
        team_id: id of this team from teams.csv file
        dm: DataManager from which this team comes from
        """
        self.team_id = team_id
        self.dm = dm

    @property
    def name(self):
        return None

    def __str__(self):
        return str(self.team_id)

    def __repr__(self):
        return str(self.team_id)
