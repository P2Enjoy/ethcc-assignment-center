from config.data import data_json
from config.inference import inference_config
from config.shifts import shifts_sites_json
from generators.demo.volunteers import test_generate_random_volunteers
from generators.inputs import generate_input_positions
from generators.shifts import generate_input_shifts

# Data de-normalization to speed inference
data_json['shifts'] = generate_input_shifts(shifts_sites_json)
data_json['positions'] = generate_input_positions(data_json)

# Only for testing
data_json['volunteers'] = test_generate_random_volunteers(data_json)

# Final compiled data
final_json = dict(inference_config, **data_json)
