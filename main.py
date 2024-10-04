# main.py

import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from espn_api.football import League

# ESPN API details
league_id = 2112580283
year = 2024
swid = "{AB1C00BE-4233-4DB5-A9F5-94D3783405F9}"
espn_s2 = "AECLN2hznzdXCpYvoAUk2gye37ETe82PqweTOBQsseH%2Bz3DNYExFgvAZwz9mGRNvrrnlVvM7Sk%2F%2BARGbmrL5MvmlHOauUR2R2lzkwX9kmODrFmEVAoxBpKL0Haa0fU753xFaViUTGbtc%2Ftzv6rjnl3ZePPp2nPMkGMEl1ueVSwgzeLrJ%2BBOehzd3SOKc9NMbKit3%2Bd5a0MQusHmqA9l9NrZJ3mfUMAPLgb3EfRh70zBDrUAKE5yZTNPtip99xEisoSGjV2tapKPEC4NOJLiQQRQw7ulWi0SG5h%2BEPsSpjSfT1A%3D%3D"

# Initialize the league
league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)

# Function to load all weekly data files with filtering
def load_weekly_data():
    data_frames = []
    # Load historical data (e.g., 2023 data)
    try:
        fantasy_data_2023 = pd.read_csv('2023/2023_ind.csv', on_bad_lines='skip', sep=',', quotechar='"')
        print(f"2023 data loaded: {len(fantasy_data_2023)} records")

        # We won't filter historical data based on injury or active status
        data_frames.append(fantasy_data_2023)
    except FileNotFoundError:
        print("2023 data file not found. Please check the file path.")
        exit()
    except pd.errors.EmptyDataError:
        print("2023 data file is empty. Please check the file content.")
        exit()

    # Load all weekly data files dynamically
    week_files = glob.glob('2024_Week_*.csv')
    week_files.sort()  # Ensure files are sorted by week number
    for file_name in week_files:
        try:
            week_data = pd.read_csv(file_name)
            if not week_data.empty:
                print(f"{file_name} loaded: {len(week_data)} records")

                # For players present in 2024 CSV files, we will filter based on injury or backup status
                week_data = filter_players(week_data)
                data_frames.append(week_data)
            else:
                print(f"{file_name} is empty. Skipping.")
        except FileNotFoundError:
            print(f"{file_name} not found. Skipping.")
        except pd.errors.EmptyDataError:
            print(f"{file_name} is empty. Skipping.")

    # Check if we have any data to proceed
    if len(data_frames) == 0:
        print("No data available for training. Exiting the script.")
        exit()

    # Combine all data
    combined_data = pd.concat(data_frames, ignore_index=True)
    return combined_data

# Function to filter out injured or backup players from 2024 data
def filter_players(df):
    # Ensure required columns are present
    required_columns = ['Name', 'injuryStatus', 'lineupSlot', 'percent_owned', 'percent_started', 'avg_points', 'injured']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: The following required columns are missing in the data: {missing_columns}")
        # Proceed without filtering based on these columns
        return df

    # Fill missing values
    df['injuryStatus'] = df['injuryStatus'].fillna('UNKNOWN')
    df['lineupSlot'] = df['lineupSlot'].fillna('BE')
    df['percent_started'] = pd.to_numeric(df['percent_started'], errors='coerce').fillna(0.0)
    df['avg_points'] = pd.to_numeric(df['avg_points'], errors='coerce').fillna(0.0)
    df['injured'] = df['injured'].fillna(False).astype(bool)

    # Exclude injured or backup players
    filtered_df = df[
        (df['injuryStatus'] == 'ACTIVE') &
        (df['injured'] == False) &
        (~df['lineupSlot'].isin(['IR', 'O', 'D', 'Q', 'BE'])) &
        (df['percent_started'] > 0.0) &
        (df['avg_points'] > 0.0)
    ]

    print(f"Records before filtering: {len(df)}")
    print(f"Records after filtering: {len(filtered_df)}")

    return filtered_df

# Function to fetch projected points and injury status for the upcoming week
def fetch_projected_points_and_injury_status(league, week):
    print(f"Fetching projected points and injury status for Week {week}...")
    projected_data = []
    # Get players on teams
    for team in league.teams:
        for player in team.roster:
            # Get projected average points
            projected_points = player.projected_avg_points if player.projected_avg_points is not None else 0.0
            # Get injury status
            injury_status = player.injuryStatus
            # Collect data
            projected_data.append({
                'Name': player.name,
                'projected_points': projected_points,
                'injuryStatus': injury_status,
                'lineupSlot': player.position
            })
    # Get free agents
    free_agents = league.free_agents(size=5000)
    for player in free_agents:
        # Get projected average points
        projected_points = player.projected_avg_points if player.projected_avg_points is not None else 0.0
        # Get injury status
        injury_status = player.injuryStatus
        # Collect data
        projected_data.append({
            'Name': player.name,
            'projected_points': projected_points,
            'injuryStatus': injury_status,
            'lineupSlot': player.position
        })
    # Create DataFrame
    projected_df = pd.DataFrame(projected_data)
    # Remove duplicates (players on rosters may also be in free agents)
    projected_df = projected_df.drop_duplicates(subset='Name', keep='first')
    return projected_df

# Prepare data for modeling
def prepare_data(combined_data):
    # Ensure 'Week' column is numeric
    combined_data['Week'] = pd.to_numeric(combined_data['Week'], errors='coerce').fillna(0).astype(int)

    # Define features and target
    features = [
        'passingCompletions', 'passingYards', 'passingTouchdowns', 'passingInterceptions',
        'rushingAttempts', 'rushingYards', 'rushingTouchdowns',
        'receivingTargets', 'receivingReceptions', 'receivingYards', 'receivingTouchdowns',
        'fumbles'
    ]

    target = 'points'

    # Check if all features exist in the data
    available_features = [feature for feature in features if feature in combined_data.columns]
    missing_features = [feature for feature in features if feature not in combined_data.columns]
    if missing_features:
        print(f"Warning: The following features are missing from the data and will be ignored: {missing_features}")
    features = available_features

    # Sort data by 'Name' and 'Week'
    combined_data = combined_data.sort_values(['Name', 'Week']).reset_index(drop=True)

    # Create lagged features
    lagged_rows = []
    grouped = combined_data.groupby('Name')

    for name, group in grouped:
        group = group.sort_values('Week').reset_index(drop=True)
        for i in range(1, len(group)):
            # Use stats from previous weeks as features
            previous_weeks_stats = group.loc[:i-1, features].mean()
            # Target is points in the current week
            target_points = group.loc[i, target]
            # Week number
            week = group.loc[i, 'Week']
            # Collect the data
            row = previous_weeks_stats.to_dict()
            row['points'] = target_points
            row['Week'] = week
            row['Name'] = name
            lagged_rows.append(row)

    # Check if we have any lagged data
    if len(lagged_rows) == 0:
        print("Not enough data to create lagged features. Exiting the script.")
        exit()

    # Create a new DataFrame from the lagged rows
    model_data = pd.DataFrame(lagged_rows)

    # Fill missing values with zeros
    model_data.fillna(0, inplace=True)

    return model_data, features, target

# Train the model
def train_model(model_data, features, target):
    X = model_data[features]
    y = model_data[target]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions and calculate MSE
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return rf_model, scaler

# Make projections for the upcoming week
def make_projections(combined_data, rf_model, scaler, features):
    # Determine the next week number
    max_week = combined_data['Week'].max()
    current_week = league.current_week
    next_week = current_week
    print(f"Making projections for Week {next_week}...")

    # Prepare data for all players for the next week
    player_projections = []
    grouped = combined_data.groupby('Name')

    for name, group in grouped:
        group = group.sort_values('Week').reset_index(drop=True)
        if len(group) >= 1:
            # Use stats up to the latest week as features
            previous_weeks_stats = group[features].mean().to_frame().T
            # Fill missing values with zeros
            previous_weeks_stats.fillna(0, inplace=True)
            # Standardize the features
            player_features_scaled = scaler.transform(previous_weeks_stats)
            # Predict next week's points
            projected_points_model = rf_model.predict(player_features_scaled)[0]
            player_projections.append({
                'Name': name,
                'Projected_Points_Model': projected_points_model
            })

    # Create a DataFrame of projections
    projections_df = pd.DataFrame(player_projections)

    # Fetch projected points and injury status for the upcoming week
    projected_data = fetch_projected_points_and_injury_status(league, next_week)

    # Merge projections with projected data
    final_projections = projections_df.merge(projected_data, on='Name', how='left')

    # Exclude players with projected points of 0 or who are injured
    final_projections = final_projections[
        (final_projections['projected_points'] > 0) &
        (final_projections['injuryStatus'] == 'ACTIVE')
    ]

    # Sort by our model's projected points
    final_projections = final_projections.sort_values('Projected_Points_Model', ascending=False).reset_index(drop=True)

    # Save projections to CSV
    final_projections.to_csv(f'Week_{next_week}_Projections.csv', index=False)
    print(f"Projections for Week {next_week} saved to 'Week_{next_week}_Projections.csv'.")

    # Print top 10 projections
    print("\nTop 10 Player Projections:")
    print(final_projections.head(25))

    # Example: Get projection for a specific player
    player_name = "Saquon Barkley"

    if player_name in final_projections['Name'].values:
        player_projection = final_projections[final_projections['Name'] == player_name]['Projected_Points_Model'].values[0]
        print(f"{player_name}'s projected points for Week {next_week}: {player_projection}")
    else:
        print(f"No projection available for {player_name}.")

if __name__ == "__main__":
    combined_data = load_weekly_data()
    model_data, features, target = prepare_data(combined_data)
    rf_model, scaler = train_model(model_data, features, target)
    make_projections(combined_data, rf_model, scaler, features)
