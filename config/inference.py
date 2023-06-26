inference_config = {
    "population_size": 100,
    "number_of_generations": 100,
    "crossover_rate": 0.85,
    "mutation_rate": 0.05,
    "tournament_size": 7,
    "select_second_best": 0.2,
    "children_of_the_atom": 0.15,
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
