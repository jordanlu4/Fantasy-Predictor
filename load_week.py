# load_week.py

from espn_api.football import League
import pandas as pd

# ESPN API details
league_id = 2112580283
year = 2024
swid = "{AB1C00BE-4233-4DB5-A9F5-94D3783405F9}"
espn_s2 = "AECLN2hznzdXCpYvoAUk2gye37ETe82PqweTOBQsseH%2Bz3DNYExFgvAZwz9mGRNvrrnlVvM7Sk%2F%2BARGbmrL5MvmlHOauUR2R2lzkwX9kmODrFmEVAoxBpKL0Haa0fU753xFaViUTGbtc%2Ftzv6rjnl3ZePPp2nPMkGMEl1ueVSwgzeLrJ%2BBOehzd3SOKc9NMbKit3%2Bd5a0MQusHmqA9l9NrZJ3mfUMAPLgb3EfRh70zBDrUAKE5yZTNPtip99xEisoSGjV2tapKPEC4NOJLiQQRQw7ulWi0SG5h%2BEPsSpjSfT1A%3D%3D"

# Initialize the league
league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)
week = 4  # Update this to the desired week number

# Function to fetch and process data for a given week
def fetch_week_data(week):
    print(f"Fetching data for Week {week}...")
    matchups = league.scoreboard(week=week)
    new_data = []
    for matchup in matchups:
        teams = [matchup.home_team, matchup.away_team]
        for team in teams:
            for player in team.roster:
                if player.stats:
                    week_stats = player.stats.get(week)  # Stats for the specified week
                    if week_stats:
                        breakdown = week_stats.get('breakdown', {})
                        player_data = {
                            'Name': player.name,
                            'Team': player.proTeam,
                            'lineupSlot': player.position,
                            'points': week_stats.get('points', 0),
                            'Week': week,
                            'passingCompletions': breakdown.get('passingCompletions', 0),
                            'passingYards': breakdown.get('passingYards', 0),
                            'passingTouchdowns': breakdown.get('passingTouchdowns', 0),
                            'passingInterceptions': breakdown.get('passingInterceptions', 0),
                            'rushingAttempts': breakdown.get('rushingAttempts', 0),
                            'rushingYards': breakdown.get('rushingYards', 0),
                            'rushingTouchdowns': breakdown.get('rushingTouchdowns', 0),
                            'fumbles': breakdown.get('fumbles', 0),
                            'receivingTargets': breakdown.get('receivingTargets', 0),
                            'receivingReceptions': breakdown.get('receivingReceptions', 0),
                            'receivingYards': breakdown.get('receivingYards', 0),
                            'receivingTouchdowns': breakdown.get('receivingTouchdowns', 0),
                            # Additional fields
                            'injuryStatus': player.injuryStatus,
                            # 'active_status': player.active,  # Removed this line
                            'percent_owned': getattr(player, 'ownershipPercentage', 0.0),
                            'percent_started': getattr(player, 'percentStarted', 0.0),
                            'avg_points': getattr(player, 'points_avg', 0.0),
                            'injured': getattr(player, 'injured', False)
                        }
                        new_data.append(player_data)
    # Convert to DataFrame and save to CSV
    week_data = pd.DataFrame(new_data)
    if not week_data.empty:
        filename = f'2024_Week_T{week}.csv'
        week_data.to_csv(filename, index=False)
        print(f"Week {week} data saved to {filename}.")
    else:
        print(f"No data available for Week {week}.")

# Call the function
if __name__ == "__main__":
    fetch_week_data(week)
