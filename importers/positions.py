import json


def generate_input_positions(input_data, organisation_data):
    # Initialize a list of positions
    positions = []

    # Create a dictionary for easy access to shifts and services
    shifts_dict = {shift['id']: shift for shift in input_data['shifts']}
    services_dict = {service['name']: service['id'] for service in input_data['services']}

    # Iterate over the organisation data
    for org in organisation_data:
        # Get the shift id and service id
        shift_id = None
        service_id = None
        for shift in shifts_dict.values():
            if shift['time_slot']['day'] in org['Day'] and shift['time_slot']['start'] in org['Shift']:
                shift_id = shift['id']
                break
        service_id = services_dict.get(org['Service'])

        # If shift id or service id is not found, raise an error
        if shift_id is None or service_id is None:
            raise ValueError('Inconsistency found in the data')

        # Create a position
        position = {
            'id': f'position_{service_id}_{shift_id}',
            'shift_id': shift_id,
            'service_id': service_id,
            'volunteers': {
                'min': int(org['Minimum']),
                'max': int(org['Maximum']),
                'recommended': int(org['Recommended'])
            }
        }

        # Add the position to the list
        positions.append(position)

    # Return the positions as a JSON string
    return positions
