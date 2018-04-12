
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc" style="margin-top: 1em;"><ul class="toc-item"><li><span><a href="#Load-Data" data-toc-modified-id="Load-Data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Load Data</a></span></li><li><span><a href="#Analyze-Data" data-toc-modified-id="Analyze-Data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Analyze Data</a></span></li></ul></div>

# In[72]:


import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# # Load Data

# In[18]:


DATA_DIR = '../data/'
TEAM_DATA_PATH = os.path.join(DATA_DIR, 'teams.csv')
REG_SEASON_DATA_PATH = os.path.join(DATA_DIR, 'RegularSeasonDetailedResults.csv')
TOURNEY_RESULTS_DATA_PATH = os.path.join(DATA_DIR, 'TourneyDetailedResults.csv')
SEEDS_DATA_PATH = os.path.join(DATA_DIR, 'TourneySeeds.csv')
SLOTS_DATA_PATH = os.path.join(DATA_DIR, 'TourneySlots.csv')


# In[5]:


teams = pd.read_csv(TEAM_DATA_PATH)


# In[7]:


print(f'There are {teams.shape[0]} teams for which we have IDs.')


# In[9]:


seasons = pd.read_csv(REG_SEASON_DATA_PATH)


# In[10]:


print(f'There are {seasons.shape[0]} rows in our reg season data file.')


# In[13]:


tourney = pd.read_csv(TOURNEY_RESULTS_DATA_PATH)


# In[14]:


print(f'There are {tourney.shape[0]} rows in our tourney info data file.')


# In[16]:


seeds = pd.read_csv(SEEDS_DATA_PATH)


# In[17]:


print(f'There are {seeds.shape[0]} rows in our seeds info data file.')


# In[19]:


slots = pd.read_csv(SLOTS_DATA_PATH)


# In[20]:


print(f'There are {slots.shape[0]} rows in our slots info data file.')


# # Analyze Data

# In[67]:


def plot_series(series_list, hist=False, title=None, xlabel=None, ylabel=None, legend=False, **kwargs):
    if not isinstance(series_list, list):
        series_list = [series_list]
    for s in series_list:
        if not hist:
            s.value_counts().sort_index().plot(kind='bar', legend=legend, **kwargs)
        else:
            s.plot(kind='hist', legend=legend, **kwargs)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.show()


# In[64]:


plot_series(seasons.Season, title='Season distribution, regular season', xlabel='Season', ylabel='Num games')


# In[65]:


plot_series(tourney.Season, title='Season distribution, tourney games!', xlabel='Season', ylabel='Num games')


# In[71]:


plot_series([seasons.Wscore, seasons.Lscore],
            hist=True,
            title='Average winning/losing team score',
            xlabel='score',
            ylabel='num games',
            legend=True,
            bins=30,
            alpha=0.8)


# In[80]:


ax = sns.distplot(seasons.Wfgm)
ax.set_title('Winning field goals made')


# In[86]:


fig = plt.figure(figsize=(15, 15))
ax = sns.heatmap(seasons.corr())
ax.set_title('Regular sesason correlations')


# In[88]:


# Row for a team with their stats
# Specific to a game
# team x year -> vector

# (team 1, year) -> a
# (team 2, year) -> b
# (a - b) -> winner
# we know the actual winner
# 1258 in 2003
select = seasons[(seasons.Season == 2003) & ((seasons.Wteam == 1258) | (seasons.Lteam == 1258))]


# In[89]:


select.shape


# In[96]:


plot_series(select[select.Wteam == 1258].Wscore, legend=True, hist=True)


# In[98]:


plot_series(select[select.Lteam == 1258].Lscore, legend=True, hist=True)

