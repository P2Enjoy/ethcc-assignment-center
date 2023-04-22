import copy
import random
from typing import Callable, Optional, Tuple


def initialize_population(services: dict, users: dict, population_size: int) -> list:
    """
    Initialize the population of assignment solutions for the genetic algorithm.

    Args:
        services (dict): A dictionary containing service constraints.
        users (dict): A dictionary containing user preferences and constraints.
        population_size (int): The number of assignment solutions to generate.

    Returns:
        list: A list of generated assignment solutions.
    """
    population = []

    # Generate population_size number of assignment solutions
    for _ in range(population_size):
        assignment_solution = {}

        for service in services.keys():
            # Randomly assign users to each service, while considering user preferences and constraints
            assigned_users = []
            for user, user_info in users.items():
                # Check if user cannot be assigned to this service
                if service not in user_info["cannot_assign"]:
                    # Assign user to service based on their preference
                    if service in user_info["preferences"]:
                        assigned_users.append(user)
                    # Assign user to service with a small probability if not in their preferences
                    elif random.random() < 0.1:
                        assigned_users.append(user)

            # Shuffle the list of assigned users to create random assignments
            random.shuffle(assigned_users)
            assignment_solution[service] = assigned_users

        # Add the generated assignment solution to the population
        population.append(assignment_solution)

    return population


def calculate_fitness(population: list, services: dict, users: dict, fitness_fn: Optional[Callable] = None) -> list:
    """
    Calculate the fitness of each assignment solution in the population.

    Args:
        population (list): A list of assignment solutions.
        services (dict): A dictionary containing service constraints.
        users (dict): A dictionary containing user preferences and constraints.
        fitness_fn (Optional[Callable]): An optional custom fitness function.

    Returns:
        list: A list of fitness scores for each assignment solution in the population.
    """
    if not fitness_fn:
        fitness_fn = default_fitness_function

    fitness_scores = []

    # Calculate the fitness score for each assignment solution in the population
    for assignment_solution in population:
        fitness_score = fitness_fn(assignment_solution, services, users)
        fitness_scores.append(fitness_score)

    return fitness_scores


def default_fitness_function(assignment_solution: dict, services: dict, users: dict) -> float:
    """
    Calculate the fitness of an assignment solution based on the criteria described in the problem statement,
    including user preferences and cannot_assign constraints.

    Args:
        assignment_solution (dict): An assignment solution to evaluate.
        services (dict): A dictionary containing service constraints.
        users (dict): A dictionary containing user preferences and constraints.

    Returns:
        float: The fitness score of the given assignment solution.
    """
    fitness = 0

    for service, assigned_users in assignment_solution.items():
        service_info = services[service]
        num_assigned_users = len(assigned_users)

        # Bonus for solutions that assign users near the recommended value
        if service_info["min"] <= num_assigned_users <= service_info["max"]:
            fitness += abs(num_assigned_users - service_info["rec"])

        # Punish solutions that assign users below the minimum value
        elif num_assigned_users < service_info["min"]:
            fitness -= (service_info["min"] - num_assigned_users) * service_info["priority"]

        # Punish solutions that assign users above the maximum value
        else:  # num_assigned_users > service_info["max"]:
            fitness -= (num_assigned_users - service_info["max"]) * service_info["priority"]

        # Punish solutions that assign users to their cannot_assign services
        for user in assigned_users:
            if service in users[user]["cannot_assign"]:
                fitness -= 100 * service_info["priority"]

        # Bonus solutions that assign users to their preferred services
        for user, user_info in users.items():
            if service in user_info["preferences"] and user in assigned_users:
                fitness += 10

    return -fitness


def selection(fitness_scores: list) -> Tuple[int, int]:
    """
    Select two parent solutions from the population based on their fitness scores.

    Args:
        fitness_scores (list): A list of fitness scores for each assignment solution in the population.

    Returns:
        Tuple[int, int]: The indices of the two selected parent solutions in the population.
    """
    # Calculate the total fitness of the population
    total_fitness = sum(fitness_scores)

    # Calculate the relative fitness of each solution
    relative_fitness = [f / total_fitness for f in fitness_scores]

    # Select the first parent using roulette wheel selection
    parent1_index = -1
    r = random.random()
    accumulator = 0
    for i, rf in enumerate(relative_fitness):
        accumulator += rf
        if accumulator >= r:
            parent1_index = i
            break

    # Select the second parent using roulette wheel selection, ensuring it's different from the first parent
    parent2_index = -1
    while parent2_index == -1 or parent2_index == parent1_index:
        r = random.random()
        accumulator = 0
        for i, rf in enumerate(relative_fitness):
            accumulator += rf
            if accumulator >= r:
                parent2_index = i
                break

    return parent1_index, parent2_index


def crossover(parent1: dict, parent2: dict) -> dict:
    """
    Combine two parent assignment solutions to create a child solution.

    Args:
        parent1 (dict): The first parent assignment solution.
        parent2 (dict): The second parent assignment solution.

    Returns:
        dict: The child assignment solution created by combining the parents.
    """
    child_solution = {}

    # Iterate over the services in the parents
    for service in parent1.keys():
        # Create two sets of users assigned to the current service in parent1 and parent2
        assigned_users_parent1 = set(parent1[service])
        assigned_users_parent2 = set(parent2[service])

        # Perform set union to combine users assigned in both parents
        combined_assigned_users = assigned_users_parent1 | assigned_users_parent2

        # Randomly assign each user from the combined set to the child solution
        child_assigned_users = []
        for user in combined_assigned_users:
            if random.random() < 0.5:
                child_assigned_users.append(user)

        child_solution[service] = child_assigned_users

    return child_solution


def mutation(solution: dict, users: dict, mutation_rate: float = 0.01) -> dict:
    """
    Mutate an assignment solution by randomly reassigning users to services.

    Args:
        solution (dict): The assignment solution to mutate.
        users (dict): A dictionary containing user preferences and constraints.
        mutation_rate (float): The probability of mutation for each user in the solution (default: 0.01).

    Returns:
        dict: The mutated assignment solution.
    """
    mutated_solution = copy.deepcopy(solution)

    # Iterate over the services in the solution
    for service, assigned_users in mutated_solution.items():
        for user in assigned_users:
            # Check if the user should be mutated based on the mutation rate
            if random.random() < mutation_rate:
                # Remove the user from the current service
                assigned_users.remove(user)

                # Find a new service for the user while considering their cannot_assign constraints
                new_service = service
                while new_service == service or new_service in users[user]["cannot_assign"]:
                    new_service = random.choice(list(mutated_solution.keys()))

                # Assign the user to the new service
                mutated_solution[new_service].append(user)

    return mutated_solution


def report_generation(generation: int, fitness_scores: list, best_solution: dict, services: dict, users: dict) -> None:
    """
    Print a report of the genetic algorithm's progress for the current generation.

    Args:
        generation (int): The current generation number.
        fitness_scores (list): The fitness scores for the current population.
        best_solution (dict): The best assignment solution found so far.
        services (dict): The input services dictionary.
        users (dict): The input users dictionary.
    """
    best_fitness = min(fitness_scores)
    worst_fitness = max(fitness_scores)
    avg_fitness = sum(fitness_scores) / len(fitness_scores)
    generation_errors = polish_errors(calculate_errors(best_solution, services, users))

    print(f"Generation {generation}:")
    print(f"  Best fitness: {best_fitness}")
    print(f"  Worst fitness: {worst_fitness}")
    print(f"  Average fitness: {avg_fitness}")
    print(f"  Best solution so far: {best_solution}")
    print(f"  Errors so far: {generation_errors}")


def calculate_errors(solution: dict, services: dict, users: dict) -> dict:
    """
    Calculate the errors in the assignment solution based on the user and service constraints.

    Args:
        solution (dict): The assignment solution to analyze.
        services (dict): The input services dictionary.
        users (dict): The input users dictionary.

    Returns:
        dict: A dictionary containing the errors for each user and service in the assignment solution.
    """
    errors = {"users": {}, "services": {}}

    # Analyze user errors
    for user, user_data in users.items():
        errors["users"][user] = {"unmet_max_assignments": False, "unmet_preference": [], "unmet_cannot_assign": []}

        user_assignments = [service for service, assigned_users in solution.items() if user in assigned_users]
        if len(user_assignments) > user_data["max_assignments"]:
            errors["users"][user]["unmet_max_assignments"] = True
            errors["users"][user]["effective_assignments"] = len(user_assignments)

        for preferred_service in user_data["preferences"]:
            if preferred_service not in user_assignments:
                errors["users"][user]["unmet_preference"].append(preferred_service)

        for cannot_assign_service in user_data["cannot_assign"]:
            if cannot_assign_service in user_assignments:
                errors["users"][user]["unmet_cannot_assign"].append(cannot_assign_service)

    # Analyze service errors
    for service, service_data in services.items():
        errors["services"][service] = {"unmet_constraint": None, "extra_users": []}

        assigned_users = solution[service]
        num_assigned_users = len(assigned_users)

        if num_assigned_users < service_data["min"]:
            errors["services"][service]["unmet_constraint"] = "min"
        elif num_assigned_users > service_data["rec"]:
            errors["services"][service]["unmet_constraint"] = "rec"
        elif num_assigned_users > service_data["max"]:
            errors["services"][service]["unmet_constraint"] = "max"
            extra_users = assigned_users[service_data["max"]:]
            errors["services"][service]["extra_users"] = extra_users

    return errors


def polish_errors(errors: dict) -> dict:
    """
    Remove users and services without unmet constraints from the errors object.

    Args:
        errors (dict): The errors object to polish.

    Returns:
        dict: A polished errors object without users and services with no unmet constraints.
    """
    polished_errors = {"users": {}, "services": {}}

    for user, user_errors in errors["users"].items():
        polished_user_errors = {}

        if user_errors["unmet_max_assignments"]:
            polished_user_errors["unmet_max_assignments"] = True

        for key, value in user_errors.items():
            if key not in ["unmet_max_assignments"] and value:
                polished_user_errors[key] = value

        if polished_user_errors:
            polished_errors["users"][user] = polished_user_errors

    for service, service_errors in errors["services"].items():
        polished_service_errors = {}

        for key, value in service_errors.items():
            if value:
                polished_service_errors[key] = value

        if polished_service_errors:
            polished_errors["services"][service] = polished_service_errors

    return polished_errors


def genetic_algorithm(services: dict, users: dict, population_size: int = 100, num_generations: int = 100,
                      mutation_rate: float = 0.01, fitness_fn: Optional[Callable] = None) -> dict:
    """
    Run the genetic algorithm to find an optimal assignment solution based on user preferences and constraints.

    Args:
        services (dict): The input services dictionary.
        users (dict): The input users dictionary.
        population_size (int): The size of the population for each generation (default: 100).
        num_generations (int): The number of generations for the genetic algorithm to run (default: 100).
        mutation_rate (float): The probability of mutation for each individual in the population (default: 0.01).
        fitness_fn (Callable, optional): An optional custom fitness function.

    Returns:
        dict: The best assignment solution found by the genetic algorithm.
    """
    # Initialize the population
    population = initialize_population(services, users, population_size)

    # If no custom fitness function is provided, use the default fitness function
    if fitness_fn is None:
        fitness_fn = default_fitness_function

    # Calculate the initial fitness scores for the population
    fitness_scores = calculate_fitness(population, services, users, fitness_fn)

    best_solution = None
    best_fitness = float('inf')

    # Main loop of the genetic algorithm
    for generation in range(num_generations):
        # Select two parent solutions based on their fitness scores
        parent1_index, parent2_index = selection(fitness_scores)

        # Create a child solution by combining the parents using crossover
        child_solution = crossover(population[parent1_index], population[parent2_index])

        # Mutate the child solution
        mutated_child_solution = mutation(child_solution, users, mutation_rate)

        # Calculate the fitness of the child solution
        child_fitness = fitness_fn(mutated_child_solution, services, users)

        # Replace the least-fit solution in the population with the child solution
        worst_fitness_index = fitness_scores.index(max(fitness_scores))
        population[worst_fitness_index] = mutated_child_solution
        fitness_scores[worst_fitness_index] = child_fitness

        # Update the best solution found so far
        if child_fitness < best_fitness:
            best_solution = mutated_child_solution
            best_fitness = child_fitness

        # Print the progress of the algorithm
        report_generation(generation, fitness_scores, best_solution, services, users)

    return best_solution
