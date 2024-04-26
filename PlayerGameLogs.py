from nba_api.stats.static import players
from nba_api.stats.static import teams
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.endpoints import playercareerstats

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# move to the directory that contains mergedenv
# type 'source mergedenv/bin/activate' to the terminal to activate environment before execution
# the environment includes nba_api.stats, pandas, numpy, sklearn, etc

def get_game_log(player_name, seasons, season_type_all_star, games_wanted = None):
    # player_name: precise spelling of the player name
    # season: 2023-2024
    # games_wanted: number of games wanted, if number exceeds all games played at designated, all games played will be shown
    # season_type_all_star: (Regular Season)|(Pre Season)|(Playoffs)|(All-Star)|(All Star)
    aggregate_df = pd.DataFrame()
    try:
        # Get player ID
        player_info = players.find_players_by_full_name(player_name)
        if not player_info:
            print(f"No player found with name {player_name}. Watch out for spelling!")
            return None
        player_id = player_info[0]['id'] 
        
        # Get player game log
        for season in seasons:
            gamelog = playergamelog.PlayerGameLog(player_id = player_id, season = season, season_type_all_star = season_type_all_star)
            gamelog_df = gamelog.get_data_frames()[0]
            aggregate_df = pd.concat([aggregate_df, gamelog_df], ignore_index=True)

        # Get the specified number of recent games
        if games_wanted:
            recent_games = aggregate_df.head(games_wanted)
        else:
            recent_games = aggregate_df

        recent_games['GAME_DATE'] = pd.to_datetime(recent_games['GAME_DATE'], format='%b %d, %Y')
        recent_games = recent_games.sort_values(by='GAME_DATE')
        
        return recent_games
    
    except Exception as e:
            print(f"An error occurred: {e}")
            return None

def against_team(team_name, dataframe):
    if dataframe is None:
        print("Dataframe is None. Cannot filter by team.")
        return None
    # get player game log against specific team
    recent_games = dataframe[dataframe['MATCHUP'].str.contains(team_name)]
    return recent_games

def cal_var(df):
    # calculate the variance of points
    avg_points = df['PTS'].mean()
    pts_var = 0
    for n in range(len(df)):
        pts_var += (df['PTS'][n] - avg_points) ** 2
    return pts_var / len(df)

# empirical probabilities measurement (Bayesian Data Analysis)
def simulator(df):
    avg_points = df['PTS'].mean()
    indicator = 0
    for n in range(1, len(df)):
        if (df['PTS'][n-1] >= avg_points and df['PTS'][n] <= avg_points) or (df['PTS'][n-1] <= avg_points and df['PTS'][n] >= avg_points):
            indicator += 1
        else:
            indicator -= 1
    return indicator

def absolute_point_diff(df):
    avg_points = df['PTS'].mean()
    point_diff = [abs(pts - avg_points) for pts in df['PTS']]
    return point_diff


if __name__ == "__main__":
    player_name = 'Jalen Duren'  # Be accurate with spelling   Typicals: Jayson Tatum, Anthony Davis, James Harden
    season = ['2023']  # '2022': 2022-2023 season
    #games_wanted = 82  # optional, parameter for get_game_log()
    season_type = "Regular Season"  # Regular Season|Pre Season|Playoffs|All-Star|All Star$

    recent_games_df = get_game_log(player_name, season, season_type) # games_wanted
    print(recent_games_df.columns)
    file_name = 'recent_games.csv'  # You can change this to your preferred path and file name
    recent_games_df.to_csv(file_name, index=False)
    #against_team_df = against_team("MEM", recent_games_df)

 
    # these two lines are used to display all rows and columns.
    # pd.set_option('display.max_columns', None) 
    # pd.set_option('display.max_rows', None)
    print(f"{player_name}'s performance:")
    print()

    if recent_games_df is not None:
        print(recent_games_df)
    #print(against_team_df)
    avg_points = recent_games_df['PTS'].mean()
    sim = simulator(recent_games_df)
    print(sim)

    # Plotting the line graph of points
    plot_df = recent_games_df.iloc[::-1]  # Reverse the DataFrame order to plot from oldest to newest game
    plt.figure(figsize=(15, 6))
    plt.plot(plot_df['GAME_DATE'], plot_df['PTS'], marker='o', label='Points')
    #plt.plot(plot_df['GAME_DATE'], plot_df['AST'], marker='o', label='Assists')
    plt.axhline(avg_points, color='r', linestyle='--', label='Average Points')  # line of average score in that season
    plt.text(plot_df['GAME_DATE'].iloc[-1], avg_points + 1, f'Avg: {avg_points:.2f}', color='r', ha='right')
    plt.title(f"{player_name}'s {season_type} Game-by-Game Performance [{season[0]}-{int(season[0])+1}]")
    plt.xlabel('Game Date')
    plt.ylabel('Points')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # Plotting the point distribution
    abs_pts_diff = absolute_point_diff(recent_games_df)
    sns.histplot(abs_pts_diff, bins=20, kde=True, color='blue', edgecolor='black', alpha=0.7)
    plt.title(f"{player_name}'s {season_type} Distribution of Absolute Point Differences[{season[0]}-{int(season[0])+1}]")
    plt.xlabel('Absolute Point Difference')
    plt.ylabel('Frequency')
    plt.show()
    
    '''
    indicator = 0
    for player in players:
        df = get_game_log(player, 2022, "Regular Season", games_wanted = None)
        indicator += simulator(df)

    print(indicator)
    '''
    
    


