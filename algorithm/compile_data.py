from config.inference import inference_config
from importers.teams import data_json_teams
from importers.services import data_json_services
from importers.shifts import read_input_shifts
from importers.positions import generate_input_positions
from importers.volunteers import read_input_volunteers

data_json = {
    'teams': data_json_teams,
    'services': data_json_services,
}

# Generating the position vacating
from importers.data import organisation_json
data_json['shifts'] = read_input_shifts(organisation_json)
data_json['positions'] = generate_input_positions(data_json, organisation_json)

# Test data
# from generators.demo.volunteers import test_generate_random_volunteers
# data_json['volunteers'] = test_generate_random_volunteers(data_json)

# Real data
data_json['volunteers'] = read_input_volunteers(data_json)

# Final compiled data
final_json = dict(inference_config, **data_json)
