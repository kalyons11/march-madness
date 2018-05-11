"""Simple test script.
"""

import sys
from functools import partial
sys.path.append('..')


from classes import *


def mean_func(dm, team, context, column, **kwargs):
    filt = dm.get_team_in_season(context.season, team)
    return dm.get_mean_stat(df=filt, team=team, column=column)


def win_pct_func(dm, team, context, **kwargs):
    # Filter on the team
    filt = dm.get_team_in_season(context.season, team)

    # Count the number of wins
    wins = filt[filt.Wteam == team.team_id]

    # Win percentage
    return wins.shape[0] / filt.shape[0]


def fg_pct_func(dm, team, context, **kwargs):
    # Take fga and fgm and divide
    fga = mean_func(dm, team, context, 'fga', **kwargs)
    fgm = mean_func(dm, team, context, 'fgm', **kwargs)
    return fgm / fga


def ft_pct_func(dm, team, context, **kwargs):
    # Take fta and ftm and divide
    fta = mean_func(dm, team, context, 'fta', **kwargs)
    ftm = mean_func(dm, team, context, 'ftm', **kwargs)
    return ftm / fta


# include the seed - bins

def parse_seed(seed_str):
    """converts 3-4 char seed string to integer value"""
    return int(seed_str[1:3])


def seed_func(dm, team, context, **kwargs):
    """return the seed of this team in the given year"""
    seeds = dm.data.seeds
    team_id = team.team_id
    yr = context.season.yr

    # try:
    this = seeds[(seeds.Season == yr) & (seeds.Team == team_id)]
    # except:
    #     print('akjlsfjslkfjlksjflksjlfkjslfj')
    #     import pdb; pdb.set_trace()

    try:
        this_seed = this.Seed.iloc[0]
    except IndexError as e:
        import pdb; pdb.set_trace()

    return parse_seed(this_seed)

# strength of schedule data


def sos_func(dm, team, context, **kwargs):
    """return the sos of this team in the given year"""
    sos = dm.data.sos
    team_id = team.team_id
    yr = context.season.yr

    this = sos[(sos.year == yr) & (sos.team_id == team_id)]

    if this.shape[0] == 0:
        return 0

    this_sos = this.sos.iloc[0]

    return this_sos


def rpi_func(dm, team, context, **kwargs):
    """return the sos of this team in the given year"""
    sos = dm.data.sos
    team_id = team.team_id
    yr = context.season.yr

    this = sos[(sos.year == yr) & (sos.team_id == team_id)]

    if this.shape[0] == 0:
        return 0

    this_rpi = this.rpi.iloc[0]

    return this_rpi


# ncaa tourney history


def tourney_games_func(dm, team, context, **kwargs):
    """return the number of tournament games in all previous years
    played by this team"""
    tour = dm.data.tourney
    me = team.team_id
    yr = context.season.yr

    this_team = tour[((tour.Wteam == me) | (tour.Lteam == me)) &
                     (tour.Season < yr)]

    return this_team.shape[0]


"""'score', 'fgm', 'fga', 'fgm3', 'fga3', 'ftm', 'fta', 'or',
            'dr', 'ast', 'to', 'stl', 'blk', 'pf',  # team"""


def prev_matchups_func(dm, team, context, **kwargs):
    # Get the other team
    me = team.team_id
    other = context.other.team_id

    # Filter back 5 years
    reg = dm.data.reg
    post = dm.data.tourney
    reg_this_yr = reg[reg.Season >= context.season.yr - 5]
    reg_this_yr = reg_this_yr.append(post)

    # Get win percentage against this team
    these = reg_this_yr[((reg.Wteam == me) & (reg.Lteam == other)) |
                        ((reg.Lteam == me) & (reg.Wteam == other))]

    if these.shape[0] > 0:
        # Get the win percentage
        wins = these[these.Wteam == me]
        return wins.shape[0] / these.shape[0]

    return 0


def main():
    # Create features
    score = Feature('score', partial(mean_func, column='score'))
    to = Feature('to', partial(mean_func, column='to'))
    blk = Feature('blk', partial(mean_func, column='blk'))
    o_reb = Feature('o_reb', partial(mean_func, column='or'))
    d_reb = Feature('d_reb', partial(mean_func, column='dr'))
    ast = Feature('ast', partial(mean_func, column='ast'))
    stl = Feature('stl', partial(mean_func, column='stl'))
    win_pct = Feature('win_pct', win_pct_func)
    fg_pct = Feature('fg_pct', fg_pct_func)
    ft_pct = Feature('ft_pct', ft_pct_func)
    prev_matchups = Feature('prev_matchups', prev_matchups_func)
    seed = Feature('seed', seed_func)
    tourney_games = Feature('tourney_games', tourney_games_func)
    sos = Feature('sos', sos_func)
    rpi = Feature('rpi', rpi_func)

    # Create model
    m = Model('base',
              [score,
               to,
               blk,
               o_reb,
               d_reb,
               win_pct,
               prev_matchups,
               fg_pct,
               ft_pct,
               ast,
               stl,
               seed,
               tourney_games,
               sos,
               rpi],
              'LogisticRegression')

    # Create trainer and run
    t = Trainer(m)
    result = t.run()

    print(result)

    print("Features used:", m.features)


if __name__ == '__main__':
    main()
