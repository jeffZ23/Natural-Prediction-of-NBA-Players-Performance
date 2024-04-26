import PlayerGameLogs


def get_player_data(player_name, seasons, season_type):
    data = PlayerGameLogs.get_game_log(player_name, seasons, season_type, games_wanted=None)

    # Data processing and feature engineering
    data['HOME'] = data['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    data['OPPONENT'] = data['MATCHUP'].apply(lambda x: x.split(' ')[-1])
    grouped = data.groupby('SEASON_ID')['PTS']
    data['AVG_PTS'] = grouped.transform(lambda x: x.shift().expanding().mean())
    data['TARGET'] = (data['PTS'] > data['AVG_PTS']).astype(int)
    data['AVG_PTS_VS_OPPONENT'] = data.groupby('OPPONENT')['PTS'].transform('mean')
    data['RECENT_FORM'] = data.groupby('SEASON_ID')['PTS'].transform(lambda x: x.rolling(window=5).mean().shift())
    data.dropna(subset=['AVG_PTS', 'RECENT_FORM'], inplace=True)        

    features = ['HOME', 'AVG_PTS_VS_OPPONENT', 'RECENT_FORM']#, 'MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'TOV']
    
    X = data[features] 
    y = data['TARGET']
    '''
    print(data[features])
    print(data['TARGET'])
    print(data['AVG_PTS'])
    '''
    return X, y