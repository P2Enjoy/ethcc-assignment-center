from config.inference import inference_config
from importers.data import data_json, organisation_json
from importers.teams import data_json_teams
from importers.services import data_json_services
from importers.shifts import data_json_shifts
from importers.positions import generate_input_positions

data_json['teams'] = data_json_teams
data_json['services'] = data_json_services
data_json['shifts'] = data_json_shifts

# Generating the position vacating
data_json['positions'] = generate_input_positions(data_json, organisation_json)



# Final compiled data
final_json = dict(inference_config, **data_json)