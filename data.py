from espn_api.football import League
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


league_id = 2112580283
year = 2024

swid = "{AB1C00BE-4233-4DB5-A9F5-94D3783405F9}"
espn_s2 = "AECLN2hznzdXCpYvoAUk2gye37ETe82PqweTOBQsseH%2Bz3DNYExFgvAZwz9mGRNvrrnlVvM7Sk%2F%2BARGbmrL5MvmlHOauUR2R2lzkwX9kmODrFmEVAoxBpKL0Haa0fU753xFaViUTGbtc%2Ftzv6rjnl3ZePPp2nPMkGMEl1ueVSwgzeLrJ%2BBOehzd3SOKc9NMbKit3%2Bd5a0MQusHmqA9l9NrZJ3mfUMAPLgb3EfRh70zBDrUAKE5yZTNPtip99xEisoSGjV2tapKPEC4NOJLiQQRQw7ulWi0SG5h%2BEPsSpjSfT1A%3D%3D"
league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)
matchups = league.scoreboard(week=1)


# Loop through each matchup and search for Travis Kelce
# for matchup in matchups:
#     for player in matchup.home_team.roster:
#         if player.name == "Josh Allen":
#             print(player.__dict__)  # Print all attributes of the player object to inpsect them

#     for player in matchup.away_team.roster:
#         if player.name == "Josh Allen":
#             print(player.__dict__)

# for matchup in matchups:
#     for player in matchup.home_team.roster:
#         if player.name == "Cooper Kupp":
#             print(f"Found Cooper Kupp in home team")

#             # Extract both projected points and actual points from the correct location
#             projected_points = player.stats[1]['projected_points']  # Projected points for Week 1
#             actual_points = player.stats[1]['points']  # Actual points scored in Week 1

#             print(f"Projected Points for Week 1: {projected_points}")
#             print(f"Actual Points Scored in Week 1: {actual_points}")

#     for player in matchup.away_team.roster:
#         if player.name == "Cooper Kupp":
#             print(f"Found Cooper Kupp in away team")

#             # Extract both projected points and actual points from the correct location
#             projected_points = player.stats[1]['projected_points']  # Projected points for Week 1
#             actual_points = player.stats[1]['points']  # Actual points scored in Week 1

#             print(f"Projected Points for Week 1: {projected_points}")
#             print(f"Actual Points Scored in Week 1: {actual_points}")



# Load the CSV file
file_path = '2023.csv'
fantasy_data = pd.read_csv(file_path)

# Scale key features
fantasy_data['receivingTargets'] *= 10
fantasy_data['receivingReceptions'] *= 10
fantasy_data['passingTouchdowns'] *= 10
fantasy_data['rushingTouchdowns'] *= 10
fantasy_data['receivingTouchdowns'] *= 10
fantasy_data['passingYards'] *= 10
fantasy_data['rushingYards'] *= 10
fantasy_data['receivingYards'] *= 10

# Define features for training
features = ['passingCompletions', 'passingYards', 'passingTouchdowns', 'passingInterceptions',
            'rushingAttempts', 'rushingYards', 'rushingTouchdowns',
            'receivingTargets', 'receivingReceptions', 'receivingYards', 'receivingTouchdowns', 'fumbles']

# Define the target variable (points scored)
target = 'points'

# Label encode the 'POS' column if necessary (optional)
# Only if you plan to use 'POS' in the model, you can include it in 'features' after encoding
label_encoder = LabelEncoder()
fantasy_data['lineupSlot'] = label_encoder.fit_transform(fantasy_data['lineupSlot'])

# Split the data into input (X) and target (y)
X = fantasy_data[features]
y = fantasy_data[target]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Feature importance
feature_importances = rf_model.feature_importances_
# for feature, importance in zip(features, feature_importances):
#     print(f"{feature}: {importance}")

new_data = []
for matchup in matchups:
    for player in matchup.home_team.roster + matchup.away_team.roster:  # Loop through both teams
        if player.stats:
            week_stats = player.stats.get(1)  # Week 1 stats
            if week_stats:
                # Calculate average points (you can adjust this logic based on available data)
                avg_points = week_stats.get('points', 0)  # If average points are not available, you might compute it later
                gp = 1  # Set to 1 as it's Week 1 (Games Played)

                # Extract relevant stats and append to new_data list
                player_data = {
                    'Name': player.name,
                    'Team': player.proTeam,  # NFL team abbreviation
                    'lineupSlot': player.position,  # Player position or lineup slot
                    'points': week_stats.get('points', 0),
                    'avg_points': avg_points,  # Points per game
                    'GP': gp,  # Games played
                    'passingCompletions': week_stats['breakdown'].get('passingCompletions', 0),
                    'passingYards': week_stats['breakdown'].get('passingYards', 0),
                    'passingTouchdowns': week_stats['breakdown'].get('passingTouchdowns', 0),
                    'passingInterceptions': week_stats['breakdown'].get('passingInterceptions', 0),
                    'rushingAttempts': week_stats['breakdown'].get('rushingAttempts', 0),
                    'rushingYards': week_stats['breakdown'].get('rushingYards', 0),
                    'rushingTouchdowns': week_stats['breakdown'].get('rushingTouchdowns', 0),
                    'fumbles': week_stats['breakdown'].get('fumbles', 0),
                    'receivingTargets': week_stats['breakdown'].get('receivingTargets', 0),
                    'receivingReceptions': week_stats['breakdown'].get('receivingReceptions', 0),
                    'receivingYards': week_stats['breakdown'].get('receivingYards', 0),
                    'receivingTouchdowns': week_stats['breakdown'].get('receivingTouchdowns', 0)
                }
                new_data.append(player_data)

# Create a DataFrame with the new data
new_data_df = pd.DataFrame(new_data)

# Save the new Week 1 data to a CSV file with the correct structure
new_data_df.to_csv('2024_Week_1.csv', index=False)

# print("2024 Week 1 data saved to 2024_Week_1.csv")


josh_allen_data = None

for matchup in matchups:
    for player in matchup.home_team.roster + matchup.away_team.roster:  # Loop through both teams
        if player.name == "CeeDee Lamb" and player.stats:
            week_1_stats = player.stats.get(1)  # Week 1 stats
            if week_1_stats:
                josh_allen_data = {
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

# If we have Josh Allen's Week 1 data, we can now predict Week 2 using the trained model
if josh_allen_data:
    # Convert to DataFrame to match the model's expected input format
    josh_allen_df = pd.DataFrame([josh_allen_data])

    # Ensure the data is scaled similarly to how the training data was scaled
    josh_allen_df['receivingTargets'] *= 10
    josh_allen_df['receivingReceptions'] *= 10
    josh_allen_df['passingTouchdowns'] *= 10
    josh_allen_df['rushingTouchdowns'] *= 10
    josh_allen_df['receivingTouchdowns'] *= 10
    josh_allen_df['passingYards'] *= 10
    josh_allen_df['rushingYards'] *= 10
    josh_allen_df['receivingYards'] *= 10

    # Predict Josh Allen's Week 2 points using the trained RandomForest model
    josh_allen_week2_projection = rf_model.predict(josh_allen_df)
    print(f"Josh Allen's 2024 Week 2 projected points: {josh_allen_week2_projection[0]}")
else:
    print("Josh Allen's Week 1 stats not found.")

 