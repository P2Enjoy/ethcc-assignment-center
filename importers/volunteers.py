import pandas as pd
import datetime

from config.inference import VOLUNTEERS


def read_input_volunteers(data_json):
    # Create a mapping of skills to their corresponding CSV columns
    skill_mapping = {
        'Speaks English': 'I can carry a conversation in English.',
        'People person': 'I am a people person :)',
        'Can say NO': 'I am comfortable saying no to people when necessary.',
        'Strong': 'I am fairly strong.',
        'Techie person': 'I am fairly tech-savvy.'
    }

    # Create a mapping of shifts to their corresponding datetime
    shift_mapping = {shift['id']: shift['time_slot'] for shift in data_json['shifts']}

    # Initialize the list of volunteers
    volunteers = []

    # Download and read the CSV data
    df = pd.read_csv(VOLUNTEERS)

    # Process each row in the CSV
    for index, row in df.iterrows():
        # Initialize the volunteer data
        volunteer = {
            'id': f"{row[0 ]} ({row[1]})",
            'team': None,
            'preferred_shifts': [],
            'avoid_shifts': []
        }

        # Create a mapping of shifts to their corresponding datetime
        shift_mapping = {
            shift['id']: f"{shift['time_slot']['day']}\n{shift['time_slot']['start']} - {shift['time_slot']['end']}"
            for shift in data_json['shifts']
        }

        # Process each skill
        volunteer_skills = []
        for skill, column in skill_mapping.items():
            if row[column].lower() == 'yes':
                volunteer_skills.append(skill)
        volunteer['skills'] = {skill: 1 for skill in volunteer_skills}

        # Process each shift
        for shift_id, shift_time in shift_mapping.items():
            if shift_time in df.columns:
                if str(row[shift_time]).lower() in ['yes', 'ok', 'x']:
                    volunteer['preferred_shifts'].append(shift_id)
                elif str(row[shift_time]).lower() in ['nope', 'no']:
                    volunteer['avoid_shifts'].append(shift_id)
            else:
                print(f"Shift {shift_time} was not addressed in the volunteer scheduled preferences")

        # Add the volunteer to the list
        volunteers.append(volunteer)

    # Return the JSON array of volunteers
    return volunteers
