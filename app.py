# Print the merged input JSON

from algorithm.compile_data import final_json
from algorithm.genetic_algorithm import genetic_algorithm, default_fitness_function, flash_generation

# print(json.dumps(final_json, indent=4))

# Run the genetic algorithm
# result = flash_generation(final_json, default_fitness_function)
result = genetic_algorithm(final_json, default_fitness_function)
# `result` will be a list of dictionaries with the optimal volunteer assignment schedule
