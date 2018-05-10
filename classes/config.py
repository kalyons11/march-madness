class Config:
    """Key configurations for the project."""
    TEAMS_FNAME = 'teams.csv'
    REG_FNAME = 'RegularSeasonDetailedResults.csv'
    TOURNEY_FNAME = 'TourneyDetailedResults.csv'
    SEEDS_FNAME = 'TourneySeeds.csv'
    SLOTS_FNAME = 'TourneySlots.csv'
    SLOTS_FNAME = 'rpi_and_sos.csv'

    DEFAULT_DATA_DIR = '../data/'

    PREDICTION_COLUMNS = ['game_id', 'Prediction']
