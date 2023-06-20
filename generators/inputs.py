import random


def generate_input_positions(input_json):
    # Initialize an empty list to store positions
    positions = []
    # Iterate over services
    for service in input_json['services']:
        service_id = service['id']

        # Iterate over shifts
        for shift in input_json['shifts']:
            shift_id = shift['id']

            # Create a position JSON object
            min_v = random.randint(1, 3)
            max_v = min_v + random.randint(1, 3)
            rec_v = random.randint(min_v, max_v)
            position = {
                'id': f'position_{service_id}_{shift_id}',
                'shift_id': shift_id,
                'service_id': service_id,
                'volunteers': {
                    'min': min_v,
                    'max': max_v,
                    'recommended': rec_v
                }
            }

            # Append the position to the list
            positions.append(position)

    return positions
