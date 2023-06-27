import csv

import requests

from config.inference import MDM, CDB

data_json = {
    "volunteers": [
        {
            "id": "v1",
            "skills": {
                "Strong": 3,
                "English": 4
            },
            "team": None,
            "preferred_shifts": ["shift_1"],
            "avoid_shifts": ["s2"]
        },
        {
            "id": "v2",
            "skills": {
                "Organized": 4,
                "French": 3
            },
            "team": None,
            "preferred_shifts": ["shift_3", "shift_4"],
            "avoid_shifts": []
        },
        {
            "id": "v3",
            "skills": {
                "Strong": 2,
                "Spanish": 4
            },
            "team": None,
            "preferred_shifts": ["shift_2"],
            "avoid_shifts": ["shift_3"]
        },
        {
            "id": "v4",
            "skills": {
                "Organized": 4,
                "English": 2
            },
            "team": None,
            "preferred_shifts": ["shift_5"],
            "avoid_shifts": ["shift_1"]
        }
    ]
}


def site_shifts_csv_to_json(site, url):
    response = requests.get(url)
    data = response.content.decode('utf-8').splitlines()
    reader = csv.reader(data)
    rows = list(reader)

    # Extract headers
    days = rows[0][1::]
    shifts = rows[1][1::]
    roles = [row[0] for row in rows[2:]]

    # Prepare JSON structure
    json_data = []
    for i, role in enumerate(roles):
        for j in range(len(days)):
            rec = rows[i + 2][j + 1]

            # Exclude disabled shifts (aka: requires "0" volunteers)
            if int(rec) > 0:
                json_data.append({
                    "Site": site,
                    "Day": days[j],
                    "Shift": shifts[j],
                    "Service": role,
                    "Minimum": max(1, int(rec) - 1),
                    "Recommended": rec,
                    "Maximum": int(rec) + 1
                })

    return json_data


# Read the master organisation data and store in a globally available variable
_mdm = site_shifts_csv_to_json("MDM", MDM)
_cdb = site_shifts_csv_to_json("CDB", CDB)
organisation_json = (_mdm + _cdb)
