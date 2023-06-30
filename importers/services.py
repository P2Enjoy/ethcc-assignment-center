import pandas as pd

from config.inference import SERVICES


def read_input_services():
    data = pd.read_csv(SERVICES)

    skills = data.columns[2:]
    services = []

    for i, row in data.iterrows():
        service = {"id": "service" + str(i + 1), "name": f"{row[0]} {row[1]}", "key_skills": list(skills[row[2:] == 1])}
        services.append(service)

    return services


# Usage
data_json_services = read_input_services()
