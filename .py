import requests

# Example URL to get historical player stats
url = "https://api.fantasydata.net/v3/nfl/stats/JSON/PlayerGameStatsByPlayerID/{season}/{week}/{playerid}"
headers = {'Ocp-Apim-Subscription-Key': 'your_api_key'}

response = requests.get(url, headers=headers)
data = response.json()
