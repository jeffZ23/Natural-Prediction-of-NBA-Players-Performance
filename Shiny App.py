from matplotlib import pyplot as plt
from shiny import App, render, ui, reactive

import PlayerGameLogs
import KNN

# type shiny run --reload app.py to run 
# paste the provided address to Google

nba_teams = [
    'ATL',  # Atlanta Hawks
    'BOS',  # Boston Celtics
    'BKN',  # Brooklyn Nets
    'CHA',  # Charlotte Hornets
    'CHI',  # Chicago Bulls
    'CLE',  # Cleveland Cavaliers
    'DAL',  # Dallas Mavericks
    'DEN',  # Denver Nuggets
    'DET',  # Detroit Pistons
    'GSW',  # Golden State Warriors
    'HOU',  # Houston Rockets
    'IND',  # Indiana Pacers
    'LAC',  # Los Angeles Clippers
    'LAL',  # Los Angeles Lakers
    'MEM',  # Memphis Grizzlies
    'MIA',  # Miami Heat
    'MIL',  # Milwaukee Bucks
    'MIN',  # Minnesota Timberwolves
    'NOP',  # New Orleans Pelicans
    'NYK',  # New York Knicks
    'OKC',  # Oklahoma City Thunder
    'ORL',  # Orlando Magic
    'PHI',  # Philadelphia 76ers
    'PHX',  # Phoenix Suns
    'POR',  # Portland Trail Blazers
    'SAC',  # Sacramento Kings
    'SAS',  # San Antonio Spurs
    'TOR',  # Toronto Raptors
    'UTA',  # Utah Jazz
    'WAS'   # Washington Wizards
]


app_ui = ui.page_fluid(
    ui.input_text("player_name", "Player Name", placeholder="Enter name"),
    ui.input_slider("seasons", "Select seasons to train the model", value=(2019, 2021), min=2017, max=2023),
    ui.input_select("game_type", "Game Type", choices=["Regular Season", "Playoffs"]),
    ui.input_select("home", "Home Game", choices=[True, False]),
    ui.input_select("opponent", "Opponent Team", choices=nba_teams),
    ui.input_action_button("submit_button", "Submit"),
    ui.output_plot("player_performance_plots"),
    ui.output_text_verbatim("player_performance"),
    ui.output_text_verbatim("model_evaluation"),
)

'''
def plot_player_performance(player_name, season, game_type):
    plot_df = PlayerGameLogs.get_game_log(player_name, [season], game_type).iloc[::-1]
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(plot_df['GAME_DATE'], plot_df['PTS'], marker='o', label='Points')
    avg_points = plot_df['PTS'].mean()
    ax.axhline(avg_points, color='r', linestyle='--', label='Average Points')
    ax.set_title(f"{player_name}'s {game_type} Game-by-Game Performance [{season}-{int(season)+1}]")
    ax.set_xlabel('Game Date')
    ax.set_ylabel('Points')
    ax.legend()
    return fig
'''
def server(input, output, session):
    submit_pressed = reactive.Value(False)
    @reactive.Effect
    @reactive.event(input.submit_button)
    def _():
        submit_pressed.set(True)

    @output
    @render.plot
    def player_performance_plots():
        player_name = input.player_name()
        seasons_range = input.seasons()
        game_type = input.game_type()

        # Generate a list of season strings
        seasons = [str(year) for year in range(seasons_range[0], seasons_range[1] + 1)]

        # Create a subplot for each season within a single figure
        fig, axs = plt.subplots(len(seasons), 1, figsize=(10, 5 * len(seasons)))
        if len(seasons) == 1:
            axs = [axs]

        for i, season in enumerate(seasons):
            plot_df = PlayerGameLogs.get_game_log(player_name, [season], game_type).iloc[::-1]
            axs[i].plot(plot_df['GAME_DATE'], plot_df['PTS'], marker='o', label='Points')
            avg_points = plot_df['PTS'].mean()
            axs[i].axhline(avg_points, color='r', linestyle='--', label='Average Points')
            axs[i].set_title(f"{player_name}'s {game_type} Performance in Season {season}")
            axs[i].set_xlabel('Game Date')
            axs[i].set_ylabel('Points')
            axs[i].legend()

        return fig
    

    @output
    @render.text
    def model_evaluation():
        if submit_pressed():
            # Reset the button click state to allow for future updates
            submit_pressed.set(False)
        player_name = input.player_name()
        seasons = input.seasons()
        season_type = input.game_type()
        X, y = KNN.get_player_data(player_name, seasons, season_type)
        _, _, _, metrics = KNN.train_evaluate_knn(X, y)
        formatted_metrics = "Performance Metrics:\n"
        for key, value in metrics.items():
            formatted_metrics += f"{key}:\n{value}\n"
        
        return formatted_metrics

    @output
    @render.text
    def player_performance():
        if submit_pressed():
            # Reset the button click state to allow for future updates
            submit_pressed.set(False)
        player_name = input.player_name()
        seasons = input.seasons()
        season_type = input.game_type()
        opponent_team = input.opponent()
        Home_court = input.home()
        avg_pts = KNN.calculate_historical_avg_points(player_name, seasons, season_type)
        pred = KNN.predict_performance(player_name, opponent_team, Home_court, seasons, season_type)
        
        return f"Predicted Performance: {'Above' if pred == 1 else 'Below'} {avg_pts} points"

app = App(app_ui, server)

'''
    def player_performance_plots():
        player_name = input.player_name()
        seasons_range = input.seasons()
        game_type = input.game_type()

        # Generate a list of season strings
        seasons = [str(year) for year in range(seasons_range[0], seasons_range[1] + 1)]

        # Create a subplot for each season within a single figure
        fig, axs = plt.subplots(len(seasons), 1, figsize=(10, 5 * len(seasons)))
        if len(seasons) == 1:
            axs = [axs]

        for i, season in enumerate(seasons):
            plot_df = PlayerGameLogs.get_game_log(player_name, [season], game_type).iloc[::-1]
            axs[i].plot(plot_df['GAME_DATE'], plot_df['PTS'], marker='o', label='Points')
            avg_points = plot_df['PTS'].mean()
            axs[i].axhline(avg_points, color='r', linestyle='--', label='Average Points')
            axs[i].set_title(f"{player_name}'s {game_type} Performance in Season {season}")
            axs[i].set_xlabel('Game Date')
            axs[i].set_ylabel('Points')
            axs[i].legend()

        return fig
    '''

