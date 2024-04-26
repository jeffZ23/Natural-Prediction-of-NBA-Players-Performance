import pandas as pd
import PlayerGameLogs
import KNN
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Assuming X and y are your features and labels respectively
# Split the dataset into training and testing sets


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


def train_evaluate_svm(X, y, test_size=0.2, random_state=42, C=1.0, kernel='linear', evaluate = True):
    """
    Train and evaluate a Support Vector Machine model on the provided dataset.

    Parameters:
    - X: Feature matrix
    - y: Target vector
    - test_size: Proportion of the dataset to include in the test split
    - random_state: Controls the shuffling applied to the data before applying the split
    - C: Regularization parameter. The strength of the regularization is inversely proportional to C
    - kernel: Specifies the kernel type to be used in the algorithm

    Returns:
    - None
    """
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the SVM classifier
    svm = SVC(C=C, kernel=kernel)

    # Train the SVM classifier
    svm.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = svm.predict(X_test_scaled)

    # Evaluation metrics
    metrics = {}
    if evaluate:
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['classification_report'] = classification_report(y_test, y_pred, zero_division=1)

    return svm, scaler, y_pred, metrics

def predict_performance(player_name, opponent, is_home_game, seasons, season_type):
    # Train the model
    X, y = get_player_data(player_name, seasons, season_type)
    model, scaler, predictions, metrics = train_evaluate_svm(X, y)

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

if __name__ == "__main__":
    player_name = "Kawhi Leonard"
    seasons =  ['2022', '2023']
    season_type = "Regular Season"
    Home_court = False
    opponent_team = "MIN"

    pred = predict_performance(player_name, opponent_team, Home_court, seasons, season_type)
    avg_pts = KNN.calculate_historical_avg_points(player_name, seasons, season_type)

    print(f"Prediction for {player_name} against {opponent_team} in the {season_type} of {seasons[-1]}")
    print(f"Home Court Advantage: {'Yes' if Home_court else 'No'}")
    print(f"Predicted Performance: {'Above' if pred == 1 else 'Below'} {avg_pts} points")