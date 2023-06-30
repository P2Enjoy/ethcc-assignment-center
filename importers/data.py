import csv

import requests

from config.inference import ORGANIZATION


def site_shifts_csv_to_json(url):
    response = requests.get(url)
    data = response.content.decode('utf-8').splitlines()
    reader = csv.reader(data)
    rows = list(reader)

    # Extract headers
    table_header_size = 2
    days = rows[0][table_header_size::]
    shifts = rows[1][table_header_size::]
    roles = [row[0] for row in rows[table_header_size:]]
    sites = [row[1] for row in rows[table_header_size:]]

    # Prepare JSON structure
    json_data = []
    for line, role in enumerate(roles):
        for column in range(len(days)):
            rec = rows[line + table_header_size][column + table_header_size]
            # Exclude disabled shifts (aka: requires "0" volunteers)
            if int(rec) > 0:
                site = sites[line]
                json_data.append({
                    "Site": site,
                    "Day": days[column],
                    "Shift": shifts[column],
                    "Service": f"{role} {site}",
                    "Minimum": max(1, int(rec) - 1),
                    "Recommended": rec,
                    "Maximum": int(rec) + 1
                })

    return json_data


# Read the master organisation data and store in a globally available variable
organisation_json = site_shifts_csv_to_json(ORGANIZATION)
