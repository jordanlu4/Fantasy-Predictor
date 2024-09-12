import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the 2023 combined CSV file
fantasy_data_2023 = pd.read_csv('2023/2023_ind.csv', on_bad_lines='skip')

# Ensure 'lineupSlot' contains only strings to avoid mixed types
fantasy_data_2023['lineupSlot'] = fantasy_data_2023['lineupSlot'].astype(str)

# Standardize player names: Remove leading/trailing whitespace and make names consistent
fantasy_data_2023['Name'] = fantasy_data_2023['Name'].str.strip()

# Debug: Print out the unique player names in the data to verify
print("Unique player names in the data:")
print(fantasy_data_2023['Name'].unique())

# Now search for Cooper Kupp's data (strictly using 2023 data)
player_name = "Josh Allen"

# Search for the player's data
player_data = fantasy_data_2023[fantasy_data_2023['Name'].str.lower() == player_name.lower()]

# Verify if the player's data is available
if player_data.empty:
    print(f"No data found for {player_name} in 2023.")
else:
    print(f"Data found for {player_name} in 2023.")

# Define features for training
features = ['passingCompletions', 'passingYards', 'passingTouchdowns', 'passingInterceptions',
            'rushingAttempts', 'rushingYards', 'rushingTouchdowns',
            'receivingTargets', 'receivingReceptions', 'receivingYards', 'receivingTouchdowns', 'fumbles']

# Define the target variable (points scored)
target = 'points'

# Filter out rows where all features are zero (players who did not play in a particular week)
filtered_data = fantasy_data_2023[(fantasy_data_2023[features].T != 0).any()]

# Split the data into input (X) and target (y)
X = filtered_data[features]
y = filtered_data[target]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Predict Cooper Kupp's Week 1 points using his 2023 data
if not player_data.empty:
    # Select the most recent week’s data for the player (last row)
    player_last_week = player_data[features].iloc[-1]

    # Reshape the data to feed into the model
    player_df = pd.DataFrame([player_last_week])

    # Predict his Week 1 points
    player_week1_projection = rf_model.predict(player_df)
    print(f"{player_name}'s projected points for 2024 Week 1 (based on 2023 data): {player_week1_projection[0]}")
else:
    print(f"No data found for {player_name} to make a prediction.")
