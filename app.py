# Print the merged input JSON
import json

from algorithm.compile_data import final_json
from algorithm.genetic_algorithm import genetic_algorithm

print(json.dumps(final_json, indent=4))

# Run the genetic algorithm
result = genetic_algorithm(final_json)
# `result` will be a list of dictionaries with the optimal volunteer assignment schedule
