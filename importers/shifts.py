from importers.data import organisation_json


def parse_shift_time(shift_time):
    start_time, end_time = shift_time.split(' - ')
    return start_time, end_time


def read_input_shifts(json_data):
    shifts = []
    seen_shifts = set()
    shift_id = 1
    for entry in json_data:
        day = entry['Day']
        site = entry['Site']
        start_time, end_time = parse_shift_time(entry['Shift'])
        shift_tuple = (day, start_time, end_time, site)
        if shift_tuple not in seen_shifts:
            shift = {
                'id': f'shift_{shift_id}',
                'time_slot': {
                    'day': day,
                    'start': start_time,
                    'end': end_time
                },
                'site': site
            }
            shifts.append(shift)
            seen_shifts.add(shift_tuple)
            shift_id += 1
    return shifts


data_json_shifts = read_input_shifts(organisation_json)
