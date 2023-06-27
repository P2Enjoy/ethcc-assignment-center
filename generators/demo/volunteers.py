import random


def test_generate_random_volunteers(input_json):
    # Initialize an empty list to store volunteers
    volunteers = []

    # Get the available skills from the input JSON
    available_skills = []
    for service in input_json['services']:
        available_skills.extend(service['key_skills'])

    # Get the total recommended number of volunteers for all positions
    total_recommended_volunteers = sum(position['volunteers']['min'] for position in input_json['positions'])
    total_recommended_volunteers += sum(position['volunteers']['max'] for position in input_json['positions'])
    total_recommended_volunteers -= sum(position['volunteers']['recommended'] for position in input_json['positions'])

    # Iterate over each volunteer
    for volunteer_id in range(1, total_recommended_volunteers):
        # Randomly select volunteer attributes
        volunteer_skills = random.sample(available_skills, k=random.randint(1, len(available_skills)))

        # Determine if preferred shifts should be void
        volunteer_preferred_shifts = []
        if random.random() > 0.80:  # 85% chance to have a void shifts
            volunteer_preferred_shifts = random.sample(input_json['shifts'],
                                                       k=random.randint(0, min(len(input_json['shifts']), 2)))

        # Determine if avoided shifts should be void
        volunteer_unavailable_shifts = []
        if random.random() > 0.90:  # 95% chance to have a void shifts
            shifts_to_be_avoided = [shift for shift in input_json['shifts'] if shift not in volunteer_preferred_shifts]
            volunteer_unavailable_shifts = random.sample(shifts_to_be_avoided,
                                                         k=random.randint(0, min(len(shifts_to_be_avoided), 2)))

        # Create a volunteer JSON object
        new_volunteer = {
            'id': f'v{volunteer_id}',
            'skills': {skill: random.randint(1, 5) for skill in volunteer_skills},
            'team': None,
            'preferred_shifts': [shift['id'] for shift in volunteer_preferred_shifts],
            'avoid_shifts': [shift['id'] for shift in volunteer_unavailable_shifts]
        }

        # Append the volunteer to the list
        volunteers.append(new_volunteer)

    return volunteers
