from espn_api.football import League

# Replace with your actual league ID
league_id = 2112580283
year = 2024  # Ongoing season

# Replace with your actual SWID and S2 values
swid = "{AB1C00BE-4233-4DB5-A9F5-94D3783405F9}"
espn_s2 = "AECLN2hznzdXCpYvoAUk2gye37ETe82PqweTOBQsseH%2Bz3DNYExFgvAZwz9mGRNvrrnlVvM7Sk%2F%2BARGbmrL5MvmlHOauUR2R2lzkwX9kmODrFmEVAoxBpKL0Haa0fU753xFaViUTGbtc%2Ftzv6rjnl3ZePPp2nPMkGMEl1ueVSwgzeLrJ%2BBOehzd3SOKc9NMbKit3%2Bd5a0MQusHmqA9l9NrZJ3mfUMAPLgb3EfRh70zBDrUAKE5yZTNPtip99xEisoSGjV2tapKPEC4NOJLiQQRQw7ulWi0SG5h%2BEPsSpjSfT1A%3D%3D"

# Initialize the league with authentication cookies
league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)

# Fetch the matchups for the current week (for example, week 1)
matchups = league.scoreboard(week=2)


# Loop through each matchup and search for Travis Kelce
for matchup in matchups:
    for player in matchup.home_team.roster:
        if player.name == "A.J. Brown":
            print(f"Found Travis Kelce in home team")
            print(player.__dict__)  # Print all attributes of the player object to inspect them

    for player in matchup.away_team.roster:
        if player.name == "A.J. Brown":
            print(f"Found Travis Kelce in away team")
            print(player.__dict__)

# for matchup in matchups:
#     for player in matchup.home_team.roster:
#         if player.name == "CeeDee Lamb":
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

# import glob
# import pandas as pd

# # Load all 2023 weekly CSV files
# file_paths = glob.glob('2023/2023_*.csv')  # Path to the folder containing 2023 CSV files

# # Iterate over each file and print the first few rows
# for fp in file_paths:
#     data = pd.read_csv(fp)
#     print(f"File: {fp}")
#     print(data.head())  # Display first few rows of each file
#     print(f"Number of rows in {fp}: {data.shape[0]}")
#     print("\n")

    
