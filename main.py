from espn_api.football import League
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ESPN API details
league_id = 2112580283
year = 2024
swid = "{AB1C00BE-4233-4DB5-A9F5-94D3783405F9}"
espn_s2 = "AECLN2hznzdXCpYvoAUk2gye37ETe82PqweTOBQsseH%2Bz3DNYExFgvAZwz9mGRNvrrnlVvM7Sk%2F%2BARGbmrL5MvmlHOauUR2R2lzkwX9kmODrFmEVAoxBpKL0Haa0fU753xFaViUTGbtc%2Ftzv6rjnl3ZePPp2nPMkGMEl1ueVSwgzeLrJ%2BBOehzd3SOKc9NMbKit3%2Bd5a0MQusHmqA9l9NrZJ3mfUMAPLgb3EfRh70zBDrUAKE5yZTNPtip99xEisoSGjV2tapKPEC4NOJLiQQRQw7ulWi0SG5h%2BEPsSpjSfT1A%3D%3D"

# Fetch Week 1 2024 matchups
league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)
matchups = league.scoreboard(week=1)

# *** Load the single combined file for 2023 ***
fantasy_data_2023 = pd.read_csv('2023/2023_ind.csv', on_bad_lines='skip', sep=',', quotechar='"')

# Ensure 'lineupSlot' contains only strings in 2023 data
fantasy_data_2023['lineupSlot'] = fantasy_data_2023['lineupSlot'].astype(str)

# Load the 2024 Week 1 and Week 2 data (already exists in the code)
fantasy_data_2024_week1 = pd.read_csv('2024_Week_1.csv')
fantasy_data_2024_week2 = pd.read_csv('2024_Week_2.csv')

# Ensure 'lineupSlot' contains only strings in 2024 data
fantasy_data_2024_week1['lineupSlot'] = fantasy_data_2024_week1['lineupSlot'].astype(str)
fantasy_data_2024_week2['lineupSlot'] = fantasy_data_2024_week2['lineupSlot'].astype(str)

# Scale the data similarly to 2023
fantasy_data_2024_week1['receivingTargets'] *= 50
fantasy_data_2024_week1['receivingReceptions'] *= 80
fantasy_data_2024_week1['passingTouchdowns'] *= 100
fantasy_data_2024_week1['rushingTouchdowns'] *= 100
fantasy_data_2024_week1['receivingTouchdowns'] *= 100
fantasy_data_2024_week1['passingYards'] *= 70
fantasy_data_2024_week1['rushingYards'] *= 80
fantasy_data_2024_week1['receivingYards'] *= 70

# Do the same for Week 2
fantasy_data_2024_week2['receivingTargets'] *= 50
fantasy_data_2024_week2['receivingReceptions'] *= 80
fantasy_data_2024_week2['passingTouchdowns'] *= 100
fantasy_data_2024_week2['rushingTouchdowns'] *= 100
fantasy_data_2024_week2['receivingTouchdowns'] *= 100
fantasy_data_2024_week2['passingYards'] *= 70
fantasy_data_2024_week2['rushingYards'] *= 80
fantasy_data_2024_week2['receivingYards'] *= 70

# Combine 2023 and 2024 data
combined_data = pd.concat([fantasy_data_2023, fantasy_data_2024_week1, fantasy_data_2024_week2], ignore_index=True)

# Re-encode 'lineupSlot' using LabelEncoder
label_encoder = LabelEncoder()
combined_data['lineupSlot'] = label_encoder.fit_transform(combined_data['lineupSlot'])

# Assign weights to the data
weights_2023 = [1.0] * len(fantasy_data_2023)
weights_2024_week1 = [0.2] * len(fantasy_data_2024_week1)
weights_2024_week2 = [0.2] * len(fantasy_data_2024_week2)
weights_combined = weights_2023 + weights_2024_week1 + weights_2024_week2

# Define features for training
features = ['passingCompletions', 'passingYards', 'passingTouchdowns', 'passingInterceptions',
            'rushingAttempts', 'rushingYards', 'rushingTouchdowns',
            'receivingTargets', 'receivingReceptions', 'receivingYards', 'receivingTouchdowns', 'fumbles']

target = 'points'

# Split data into input (X) and target (y)
X = combined_data[features]
y = combined_data[target]

# Train/test split
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y, weights_combined, test_size=0.2, random_state=42
)

# Train the Random Forest model using sample weights
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train, sample_weight=weights_train)

feature_importances = rf_model.feature_importances_
# for feature, importance in zip(features, feature_importances):
    # print(f"{feature}: {importance}")

# Make predictions and evaluate
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Universal player projection
player_name = "CeeDee Lamb"
player_data = None
for matchup in matchups:
    for player in matchup.home_team.roster + matchup.away_team.roster:
        if player.name == player_name and player.stats:
            week_1_stats = player.stats.get(1)
            if week_1_stats:
                player_data = {
                    'passingCompletions': week_1_stats['breakdown'].get('passingCompletions', 0),
                    'passingYards': week_1_stats['breakdown'].get('passingYards', 0),
                    'passingTouchdowns': week_1_stats['breakdown'].get('passingTouchdowns', 0),
                    'passingInterceptions': week_1_stats['breakdown'].get('passingInterceptions', 0),
                    'rushingAttempts': week_1_stats['breakdown'].get('rushingAttempts', 0),
                    'rushingYards': week_1_stats['breakdown'].get('rushingYards', 0),
                    'rushingTouchdowns': week_1_stats['breakdown'].get('rushingTouchdowns', 0),
                    'receivingTargets': week_1_stats['breakdown'].get('receivingTargets', 0),
                    'receivingReceptions': week_1_stats['breakdown'].get('receivingReceptions', 0),
                    'receivingYards': week_1_stats['breakdown'].get('receivingYards', 0),
                    'receivingTouchdowns': week_1_stats['breakdown'].get('receivingTouchdowns', 0),
                    'fumbles': week_1_stats['breakdown'].get('fumbles', 0)
                }

# Predict player performance for Week 2
if player_data:
    player_df = pd.DataFrame([player_data])
    player_df['receivingTargets'] *= 50
    player_df['receivingReceptions'] *= 80
    player_df['passingTouchdowns'] *= 100
    player_df['rushingTouchdowns'] *= 100
    player_df['receivingTouchdowns'] *= 100
    player_df['passingYards'] *= 70
    player_df['rushingYards'] *= 80
    player_df['receivingYards'] *= 70


    # Prediction
    player_week2_projection = rf_model.predict(player_df)
    print(f"{player_name}'s 2024 Week 2 projected points: {player_week2_projection[0]}")
else:
    print(f"{player_name}'s Week 1 stats not found.")




