import os
from dotenv import load_dotenv

load_dotenv()

# Teams
TEAMS = os.getenv('TEAMS')
# Shifts per site
MDM = os.getenv('MDM')
CDB = os.getenv('CDB')
# Skills oer service
SERVICES = os.getenv('SERVICES')
# Volunteer database
VOLUNTEERS = os.getenv('VOLUNTEERS')

# Generation parameters
inference_config = {
    "population_size": 10,
    "number_of_generations": 350,
    "crossover_rate": 0.95,
    "mutation_rate": 0.95,
    "tournament_size": 7,
    "select_second_best": 0.15,
    "rare_mutation_rate": 0.75,
    "fitness_weights": {
        # OUTSTANDING
        # Criteria 1: Each position has the recommended number of volunteers at best or between min/max
        "recommended_volunteers": 1000,
        # OPTIONAL
        # Criteria 2: Each service has at least a volunteer per every key skill required by the service
        "key_skills": 50,
        # OPTIONAL
        # Criteria 3: All volunteers must not be assigned more than once per day
        "single_assignment_per_day": 10,
        # OUTSTANDING
        # Criteria 4: All volunteers must not be assigned more than 4 times total
        "limited_assignments": 250,
        # OPTIONAL
        # Criteria 5: All assignments for a position concerns volunteers of the same team as much as possible
        "team_variety": 7,
        # OPTIONAL
        # Criteria 6: All assignments respect the volunteer expression regarding their shift preferences, both to be assigned and not to be assigned
        "shift_preferences": 2,
        # OUTSTANDING
        "shift_unavailable": 500,
        # OUTSTANDING
        # Criteria 7: A volunteer can not be assigned at two different sites at the same time
        "no_multiple_sites": 2000,
        # OUTSTANDING
        # Criteria 8: There must not be a vacant position
        "no_vacant_positions": 500,
    }
}
