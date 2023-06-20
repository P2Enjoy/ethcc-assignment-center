inference_config = {
    "population_size": 200,
    "number_of_generations": 1000,
    "crossover_rate": 0.85,
    "mutation_rate": 0.05,
    "tournament_size": 30,
    "select_second_best": 0.2,
    "fitness_weights": {
        # Chaque service dispose des resources necessaires par shift
        "recommended_volunteers": 10,
        # Un service dispose des valontaires ayant l'esenseble de compétences cléfs pour le poste
        "key_skills": 8,
        # Un volontaire ne devrait pas opérer plus d'une affectation par jour
        "single_assignment_per_day": 5,
        # Un volontaire doit opérer un nombre minimum et maximum de affectations
        "limited_assignments": 4,
        # Un volontaire est affecté de préference à un team
        "team_variety": 7,
        # Un volontaire peut exprimer des préférences quant'aux tranches horaires à opérer et ne pas opérer
        "shift_preferences": 6,
        # Un volontaire ne devrait pas changer de site dans la même journée
        "no_multiple_sites": 9,
        # Il ne devrait y avoir des emplacements vides, tous les postes sont pourvus.
        "no_vacant_positions": 11      #
    }
}
