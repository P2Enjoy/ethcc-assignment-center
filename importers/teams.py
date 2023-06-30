import pandas as pd

from config.inference import TEAMS


def read_input_teams():
    df = pd.read_csv(TEAMS, sep=',')

    # Drop any empty rows or columns
    df = df.dropna()

    teams = []
    for i, row in df.iterrows():
        services = [f"{x.strip()} {row[1]}" for x in row[3].split(',')]
        team = {"name": row[0], "leader": row[2],  "services": list(services)}
        teams.append(team)

    return teams


data_json_teams = read_input_teams()
