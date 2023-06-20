def generate_input_shifts(input_json):
    # Initialize an empty list to store shifts
    shifts = []
    shift_id = 0
    # Iterate over days
    for day in input_json['days']:
        # Iterate over time slots
        for time_slot in input_json['time_slot']:
            start_time = time_slot['start']
            end_time = time_slot['end']

            # Iterate over sites
            for site in input_json['site']:
                shift_id += 1
                shift = {
                    'id': f'shift_{shift_id}',
                    'time_slot': {
                        'day' : day,
                        'start': start_time,
                        'end': end_time
                    },
                    'site': site
                }

                # Append the shift to the list
                shifts.append(shift)

    return shifts
