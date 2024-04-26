import pandas as pd
import PlayerGameLogs
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


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

def train_evaluate_knn(X, y, test_size=0.2, random_state=42, n_neighbors=5, evaluate=True):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Apply Cross-Validation on the training set
    cv_scores = cross_val_score(knn, X_train, y_train, cv=10)   # Performance Estimate
    print(f"Average Cross-Validation Score: {round(cv_scores.mean(), 2)}")

    # Train the model on the entire training set and predict on the test set
    knn.fit(X_train_scaled, y_train)  # Final Model 
    predictions = knn.predict(X_test_scaled)

    # Evaluation metrics
    metrics = {}
    if evaluate:
        metrics['accuracy'] = accuracy_score(y_test, predictions)
        metrics['classification_report'] = classification_report(y_test, predictions, zero_division=1)

    return knn, scaler, predictions, metrics



'''
X, y = get_player_data("Kawhi Leonard", ['2022', '2023'], "Regular Season")
model, scaler, predictions, metrics = train_evaluate_knn(X, y)
'''

'''
# Print the predictions
print("Predictions:")
print(predictions)

# Print the performance metrics
print("\nPerformance Metrics:")
for key, value in metrics.items():
    print(f"{key}:")
    print(value)
'''

def predict_performance(player_name, opponent, is_home_game, seasons, season_type):
    # Train the model
    X, y = get_player_data(player_name, seasons, season_type)
    model, scaler, predictions, metrics = train_evaluate_knn(X, y)

    # Retrieve historical data
    historical_data = PlayerGameLogs.get_game_log(player_name, seasons, season_type, games_wanted=None)
    historical_data['OPPONENT'] = historical_data['MATCHUP'].apply(lambda x: x.split(' ')[-1])

    # Calculate AVG_PTS_VS_OPPONENT
    avg_pts_vs_opponent = historical_data[historical_data['OPPONENT'] == opponent]['PTS'].mean()

    # Calculate RECENT_FORM - the most recent value from the rolling mean
    recent_form = historical_data['PTS'].rolling(window=5).mean().shift().iloc[-1]

    # Construct the feature vector
    feature_vector = {
        'HOME': 1 if is_home_game else 0,
        'AVG_PTS_VS_OPPONENT': avg_pts_vs_opponent,
        'RECENT_FORM': recent_form
    }

    # Convert to DataFrame
    X = pd.DataFrame([feature_vector])

    # Scale the features
    X_scaled = scaler.transform(X)

    # Make the prediction
    prediction = model.predict(X_scaled)

    # Print the performance metrics
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        if key == "accuracy":
            print(f"{key}:")
            print(round(value, 2))
        else:
            print(f"{key}:")
            print(value)    

    return prediction[0]

def calculate_historical_avg_points(player_name, seasons, season_type):
    # Calculate the historical average points for a player
    
    data = PlayerGameLogs.get_game_log(player_name, seasons, season_type, games_wanted=None)

    print(data)
    # Calculate the expanding mean of the player's points, excluding the current game's points
    data['AVG_PTS'] = data.groupby('SEASON_ID')['PTS'].transform(lambda x: x.shift().expanding().mean())

    # Return the average points for the last game
    return round(data['AVG_PTS'].iloc[-1], 1)


# calculate average accuracy of major players in the league
def calculate_average_accuracy(player_names, seasons, season_type):
    results = {}
    total_accuracy = 0
    cv_score_total = 0
    player_count = len(player_names)
    
    for player_name in player_names:
        # Assuming there's a function similar to 'get_player_data' that ends with calculating accuracy
        X, y = get_player_data(player_name, seasons, season_type)
        
        # Split the data into features and target variable, then into training and testing sets
        # This part is assumed based on common practice with scikit-learn models
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Feature scaling
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Training the KNN model
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_scaled, y_train)
        
        # Predicting and calculating accuracy
        predictions = knn.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        
        total_accuracy += accuracy

        # Cross-validation
        cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
        cv_score_mean = np.mean(cv_scores)

        cv_score_total += cv_score_mean
        
        # Store results
        results[player_name] = {'Accuracy': accuracy, 'CV Score': cv_score_mean}
        print(f'Player: {player_name}, Accuracy: {accuracy:.2f}, CV Score: {cv_score_mean:.2f}')
    
    
    # Calculate the average accuracy
    average_accuracy = total_accuracy / player_count
    average_cross_val_score = cv_score_mean / player_count
    return average_accuracy



if __name__ == "__main__":
    
    
    player_name = "Stephen Curry"
    seasons =  ['2023']
    season_type = "Regular Season"
    Home_court = False
    opponent_team = "LAC"

    pred = predict_performance(player_name, opponent_team, Home_court, seasons, season_type)
    avg_pts = calculate_historical_avg_points(player_name, seasons, season_type)

    print(f"Prediction for {player_name} against {opponent_team} in the {season_type} of {seasons[-1]}")
    print(f"Home Court Advantage: {'Yes' if Home_court else 'No'}")
    print(f"Predicted Performance: {'Above' if pred == 1 else 'Below'} {avg_pts} points")
    
    
   
    

    '''
    players_list = ["Jalen Williams", "Mikal Bridges", "James Harden", "Dejounte Murray", "Damian Lillard",
                     "Kevin Durant", "Stephen Curry", "Nikola Jokic", "Jordan Poole", "Demar DeRozan",
                    "Joel Embiid", "Kawhi Leonard", "Bradley Beal", "Karl-Anthony Towns", "Jaylen Brown",
                    "Tyrese Maxey", "Donovan Mitchell", "Bam Adebayo", "Devin Booker", "Kyrie Irving",
                    "Chet Holmgren", "Klay Thompson", "Tyrese Haliburton", "Miles Bridges", "John Collins",
                    "De'Aaron Fox", "Donte DiVincenzo", "Austin Reaves", "Franz Wagner", "CJ McCollum",
                    "Cade Cunningham", "Scottie Barnes", "Victor Wembanyama", "Jerami Grant", "Fred Vanvleet",
                    "Stephen Curry"]
    
    seasons = ["2023"]  # Example season, adjust as necessary
    season_type = "Regular Season"  # Example season type
    average_accuracy = calculate_average_accuracy(players_list, seasons, season_type)
    print(f'Average Prediction Accuracy: {average_accuracy:.2f}')
    
    '''
    
    
    
    