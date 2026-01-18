# Natural-Prediction-of-NBA-Players-Performance
Natural Prediction of NBA Players' Performance
This repository implements a machine-learning pipeline to predict whether an NBA player will score above or below their average points in a game using K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) classifiers. It features data retrieval and processing utilities built on the NBA API, and an interactive Shiny application for visualization and real-time predictions. 
github.com
raw.githubusercontent.com
raw.githubusercontent.com
raw.githubusercontent.com
raw.githubusercontent.com

Features
Data Acquisition

Retrieves historical game logs via the nba_api package. 
raw.githubusercontent.com
github.com

Provides utility functions for filtering by opponent, computing variance, and simulating form. 
raw.githubusercontent.com

Feature Engineering

Computes home/away indicators, season-to-date averages, opponent-specific averages, and recent rolling form. 
raw.githubusercontent.com

Machine Learning Models

KNN Classifier (KNN.py): trains and evaluates a K-Nearest Neighbors model with cross-validation and standardized features. 
raw.githubusercontent.com

SVM Classifier (SVM.py): implements a Support Vector Machine with configurable kernel and regularization. 
raw.githubusercontent.com

Interactive Visualization

A Shiny-based Python web app (Shiny App.py) for plotting game-by-game performance and displaying model predictions. 
raw.githubusercontent.com

Installation
Clone the repository

bash
Copy
Edit
git clone https://github.com/jeffZ23/Natural-Prediction-of-NBA-Players-Performance.git
cd Natural-Prediction-of-NBA-Players-Performance
Create and activate a virtual environment

bash
Copy
Edit
python3 -m venv env
source env/bin/activate
Install dependencies

bash
Copy
Edit
pip install pandas numpy scikit-learn nba_api matplotlib seaborn shiny requests
pandas: high-performance data structures for tabular data 
pandas.pydata.org
pandas.pydata.org

scikit-learn: machine learning library providing classification_report and other metrics 
scikit-learn.org

nba_api: client for NBA stats endpoints 
github.com

shiny for Python: interactive web UI framework

Usage
1. Data Retrieval
python
Copy
Edit
from PlayerGameLogs import get_game_log

# Fetch game logs for LeBron James in 2023 Regular Season
df = get_game_log("LeBron James", ["2023"], "Regular Season")
print(df.head())
2. Train and Evaluate KNN
python
Copy
Edit
from KNN import get_player_data, train_evaluate_knn

X, y = get_player_data("Stephen Curry", ["2023"], "Regular Season")
model, scaler, preds, metrics = train_evaluate_knn(X, y)
print(metrics["accuracy"])
print(metrics["classification_report"])
3. Train and Evaluate SVM
python
Copy
Edit
from SVM import get_player_data, train_evaluate_svm

X, y = get_player_data("Kawhi Leonard", ["2022", "2023"], "Regular Season")
svm_model, scaler, preds, metrics = train_evaluate_svm(X, y, kernel="rbf", C=0.5)
print(metrics["accuracy"])
4. Run the Shiny App
bash
Copy
Edit
python "Shiny App.py"
# Then open the provided local URL in your browser.
Project Structure
graphql
Copy
Edit
├── KNN.py               # K-Nearest Neighbors model and prediction pipeline :contentReference[oaicite:10]{index=10}
├── SVM.py               # Support Vector Machine pipeline :contentReference[oaicite:11]{index=11}
├── PlayerGameLogs.py    # Utilities for fetching and processing NBA game logs :contentReference[oaicite:12]{index=12}
├── Shiny App.py         # Shiny app UI and server logic for visualization :contentReference[oaicite:13]{index=13}
├── recent_games.csv     # Example output from PlayerGameLogs :contentReference[oaicite:14]{index=14}
└── mergedenv/ myenv/    # Virtual environment directories (ignored in `.gitignore`)
Examples
Predict a Player’s Next Game Performance

python
Copy
Edit
from KNN import predict_performance
pred = predict_performance("Kevin Durant", "LAL", True, ["2023"], "Regular Season")
print("Above average" if pred == 1 else "Below average")
Visualize Historical Performance
Launch the Shiny app, enter a player name and season range, and view game-by-game scoring plots.

Lineup and Availability Monitoring
----------------------------------
You can now fold in roster news to understand why a player might underperform (for example, when a star returns from injury and pushes a teammate to the bench) or why others could see a usage bump when starters sit.

### Quick CLI smoke test

1) Install dependencies if you have not already:

```bash
pip install pandas numpy scikit-learn nba_api matplotlib seaborn shiny requests
```

2) Pull the current roster signals for a team (e.g., ATL). This prints the depth chart, injury list, and any starter changes/returning players compared with optional snapshots:

```bash
python LineupMonitor.py --team ATL --save-snapshots
```

This saves timestamped snapshots to `snapshots/` so you can re-run later with `--prev-depth` and `--prev-injuries` to see what changed.

### Programmatic usage

```python
from LineupMonitor import (
    get_team_depth_chart,
    get_team_injuries,
    save_snapshot,
    summarize_lineup_signals,
)

# Grab today's baseline for the Hawks.
depth_chart = get_team_depth_chart("ATL")
injuries = get_team_injuries("ATL")
save_snapshot(depth_chart, "snapshots/atl-depth-today.json")
save_snapshot(injuries, "snapshots/atl-injuries-today.json")

# Run the same code tomorrow and compare to yesterday's snapshots.
signals = summarize_lineup_signals(
    "ATL",
    previous_depth_snapshot="snapshots/atl-depth-today.json",
    previous_injury_snapshot="snapshots/atl-injuries-today.json",
)

print("Returning players:\n", signals["returning_players"])  # Previously listed as out, now active
print("Starter changes:\n", signals["starter_changes"])       # New starters and who they replaced
print("Sidelined players:\n", signals["sidelined_players"])   # Currently listed as out or questionable
```

The signals align with the example case of a star returning (e.g., Trae Young reclaiming a starting spot and reducing Nickeil Alexander-Walker’s workload). Snapshots let you capture roster state before and after news breaks so you can generate model features that penalize returning-from-absence performances or boost teammates filling in for injured players.

Contributing
Fork the repository.

Create a feature branch: git checkout -b my-feature.

Commit your changes and push: git push origin my-feature.

Submit a pull request.

License
This project is released under the MIT License.
