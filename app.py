from flask import Flask, render_template, request
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Load player projections from CSV file
def load_projections():
    try:
        projections = pd.read_csv('Week_5_Projections.csv')  # Replace with the path to your projections CSV
        return projections
    except FileNotFoundError:
        print("Error: Projection file not found.")
        return None

# Home route to display player selection form
@app.route('/')
def index():
    projections = load_projections()
    if projections is not None:
        players = projections['Name'].tolist()  # Get list of player names from CSV
    else:
        players = []
    return render_template('index.html', players=players)

# Route to display player projection details
@app.route('/projection', methods=['POST'])
def show_projection():
    player_name = request.form.get('player')  # Get the player name from the form
    projections = load_projections()
    if projections is not None and player_name in projections['Name'].values:
        player_data = projections[projections['Name'] == player_name].iloc[0]
        projected_points_model = player_data['Projected_Points_Model']
        projected_points_espn = player_data['projected_points']
        injury_status = player_data['injuryStatus']
        lineup_slot = player_data['lineupSlot']
    else:
        player_data = None
        projected_points_model = None
        projected_points_espn = None
        injury_status = None
        lineup_slot = None

    return render_template('projection.html', player_name=player_name,
                           projected_points_model=projected_points_model,
                           projected_points_espn=projected_points_espn,
                           injury_status=injury_status,
                           lineup_slot=lineup_slot)

if __name__ == "__main__":
    app.run(debug=True)