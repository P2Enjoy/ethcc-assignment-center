# Teams
TEAMS = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQM-04xKkb6pffTENTjGLIsjFw2L3hTCFgMX4leZF7WyjUc1Mh3MBCZGKeIfqAfM829Ml6qEQg43x4C/pub?output=csv'
# Shifts per site
MDM = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTaKfXjEWftHavdKwD1YJKLR1noJdEi-TskhGcUTUGcQ_6O67YX9MIsTEE1WGH4P6O1hZ84-gerYSMO/pub?output=csv"
CDB = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQwrj4_i1SUflGfopYKZ8iBtz5XUSM9DtCFfvC1v-SeqGXWRTn3n-Ugf2F2IH999Zev1drD1Jws48sF/pub?output=csv"
# Skills oer service
SERVICES = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRI2YS7YoZPIgu-zXbbC7Wj3KnC_R9NgOt66CwDZCLm9M4-TCpe7vw3ikRKNXzXFsvOVQcGpbohQumC/pub?output=csv"

# Generation parameters
inference_config = {
    "population_size": 350,
    "number_of_generations": 1500,
    "crossover_rate": 0.85,
    "mutation_rate": 0.05,
    "tournament_size": 7,
    "select_second_best": 0.2,
    "rare_mutation_rate": 0.15,
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
