import copy
import json
import statistics
import random
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict

from exporters.csv import report_generation
from algorithm.models import TimeSlot, Shift, Service, Team, Volunteer, Position, Assignment


def initialize_population_assign_volunteer(assignments, position, position_volunteers, service_dict, shift, teams,
                                           volunteer, volunteer_assignment_count, volunteer_daily_assignment_count,
                                           volunteer_dict):
    assignments.append(Assignment(position_id=position.id, volunteer_id=volunteer.id))
    position_volunteers[position.id].add(volunteer.id)
    volunteer_assignment_count[volunteer.id] += 1
    volunteer_daily_assignment_count[volunteer.id][shift.time_slot.day] += 1
    # Assign the volunteer team to the generation specific volunteer during its first assignment
    if volunteer_dict[volunteer.id].team is None:
        current_service_name = service_dict[position.service_id].name
        suitable_teams = [team for team in teams if current_service_name in team.services]

        # Randomly select a suitable team
        if suitable_teams:
            selected_team = random.choice(suitable_teams)
            volunteer_dict[volunteer.id].team = selected_team.name


def initialize_population_volunteers_weighted_sample(num_volunteers, candidates, volunteer_assignment_count):
    if not candidates:
        return []

    weights = np.array([(volunteer_assignment_count[vol.id] + 1) for vol in candidates], dtype=float)
    sum_weights = weights.sum()
    # Avoid division by 0 if all counts are 0, in that case, make the weights uniform
    if sum_weights == 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights /= sum_weights  # normalize weights to sum to 1
    size = max(0, min(len(candidates), num_volunteers))
    selected_volunteers = list(np.random.choice(candidates, size=size, replace=False, p=weights))
    return selected_volunteers


def initialize_population(population_size: int, positions: List[Position], volunteers: List[Volunteer],
                          teams: List[Team], services: List[Service], shifts: List[Shift]) \
        -> List[List[List[Assignment] | List[Volunteer]]]:
    population = []

    # Create various dictionaries for easier lookup
    position_dict = {position.id: position for position in positions}
    service_dict = {service.id: service for service in services}
    shift_dict = {shift.id: shift for shift in shifts}

    for num in range(population_size):
        assignments = []
        position_volunteers = defaultdict(set)

        # Create cloned dictionaries of generation's specific volunteers for easier lookup
        gen_specific_vol = copy.deepcopy(volunteers)
        volunteer_dict = {vol.id: vol for vol in gen_specific_vol}

        # Count of assignments per volunteer for the current population
        volunteer_assignment_count = defaultdict(int)
        # Count of assignments per day for each volunteer for the current population
        volunteer_daily_assignment_count = defaultdict(lambda: defaultdict(int))

        # Assign generation specifics volunteers considering the constraints
        for position in positions:
            service = service_dict[position.service_id]
            required_skills = set(service.key_skills)
            shift = shift_dict[position.shift_id]

            # Select candidates who have required skills and
            # do not have the current position shift in their avoid list
            suitable_candidates = [vol for vol in gen_specific_vol if
                                   any(skill in vol.skills for skill in required_skills) and
                                   position.shift_id not in vol.avoid_shifts]

            # This subset have ALL the skills for the position
            perfect_candidates = [vol for vol in suitable_candidates if
                                  all(skill in vol.skills for skill in required_skills)]

            # Between suitable candidates, the one from the same team are preferred (if any)
            assigned_volunteer_ids = position_volunteers[position.id]
            assigned_teams = {volunteer_dict[vol_id].team for vol_id in assigned_volunteer_ids}
            common_team = None
            if assigned_teams:
                common_team = assigned_teams.pop()
            if common_team is not None:
                suitable_team_candidates = [vol for vol in suitable_candidates if vol.team == common_team]
            else:
                suitable_team_candidates = []

            # This subset have ALL the skills for the position
            perfect_team_candidates = [vol for vol in suitable_team_candidates if
                                       all(skill in vol.skills for skill in required_skills)]

            required_volunteers = position.min_volunteers
            assigned_volunteers = 0
            candidates_pools = [
                [perfect_team_candidates, 1],
                [suitable_team_candidates, required_volunteers],
                [perfect_candidates, required_volunteers],
                [suitable_candidates, required_volunteers],
                [[vol for vol in gen_specific_vol
                  if position.shift_id not in vol.avoid_shifts], required_volunteers]
            ]

            # Loop until there are at least the minimum required number of volunteers
            # or there are no volunteers left.
            goal_reached = False
            while not goal_reached:
                goal_reached = True

                # Loop through each pool of candidates in priority order.
                for pool, sample_num in candidates_pools:
                    # Filter available candidates from the current pool whi have less than 4 assignments,
                    # and have not reached daily assignment limit
                    available_from_pool = [vol for vol in pool if
                                           volunteer_assignment_count[vol.id] < 4 and
                                           volunteer_daily_assignment_count[vol.id][shift.time_slot.day] < 1]

                    for vol in initialize_population_volunteers_weighted_sample(sample_num, available_from_pool, volunteer_assignment_count):
                        # Assign the volunteer to the position using the provided function
                        initialize_population_assign_volunteer(assignments, position, position_volunteers,
                                                               service_dict, shift, teams, vol,
                                                               volunteer_assignment_count,
                                                               volunteer_daily_assignment_count, volunteer_dict)
                        assigned_volunteers += 1

                    if assigned_volunteers < required_volunteers:
                        goal_reached = False
                    else:
                        goal_reached = True
                        break

        # Attempt to add preferred shifts for the remaining generation specifics volunteers
        for volunteer in [vol for vol in gen_specific_vol if volunteer_assignment_count[vol.id] < 4]:
            preferred_shifts = set(volunteer.preferred_shifts) - set(
                position_dict[assignment.position_id].shift_id for assignment in assignments
                if assignment.volunteer_id == volunteer.id
            )

            for shift_id in preferred_shifts:
                if volunteer_daily_assignment_count[volunteer.id][shift_dict[shift_id].time_slot.day] < 1:
                    available_positions = [position for position in positions if position.shift_id == shift_id and
                                           len(position_volunteers[position.id]) < position.max_volunteers]

                    if available_positions:
                        selected_position = random.choice(available_positions)
                        initialize_population_assign_volunteer(assignments, selected_position, position_volunteers,
                                                               service_dict, shift_dict[shift_id],
                                                               teams, volunteer, volunteer_assignment_count,
                                                               volunteer_daily_assignment_count, volunteer_dict)

        population.append([assignments, gen_specific_vol])

        if population_size < 2:
            print(f'Rare mutation occurred.')
        else:
            print(
                f'Generated population {num} out of total pool {population_size} with diversity of {calculate_population_diversity(population[-2:])}.')

    return population


def default_fitness_function(assignments: List[Assignment], positions: List[Position], volunteers: List[Volunteer],
                             teams: List[Team], services: List[Service], shifts: List[Shift],
                             fitness_weights: Dict[str, int]) -> int:
    """
    This function evaluates the fitness of a single solution based on the criteria provided.
    """
    fitness = 0
    volunteer_assignments = {}
    position_volunteers = {}

    # Create dictionaries for easier lookup
    volunteer_dict = {volunteer.id: volunteer for volunteer in volunteers}
    position_dict = {position.id: position for position in positions}
    service_dict = {service.id: service for service in services}
    shift_dict = {shift.id: shift for shift in shifts}

    # Count assignments per volunteer and per position
    for assignment in assignments:
        volunteer_assignments.setdefault(assignment.volunteer_id, []).append(assignment)
        position_volunteers.setdefault(assignment.position_id, []).append(assignment.volunteer_id)

    # OUTSTANDING
    # Criteria 1: Each position has the recommended number of volunteers at best or between min/max
    for position_id, volunteer_ids in position_volunteers.items():
        volunteer_tot = len(volunteer_ids)
        if position_dict[position_id].min_volunteers <= volunteer_tot <= position_dict[position_id].max_volunteers:
            fitness += fitness_weights["recommended_volunteers"] - abs(
                volunteer_tot - position_dict[position_id].recommended_volunteers)
        else:
            fitness -= fitness_weights["recommended_volunteers"]

    # OPTIONAL
    # Criteria 2: Each service has at least a volunteer per every key skill required by the service
    for position in positions:
        service = service_dict[position.service_id]
        volunteers_in_position = [volunteer_dict[vol_id] for vol_id in position_volunteers.get(position.id, [])]
        skills = set()
        for v in volunteers_in_position:
            skills.update(v.skills.keys())
        if all(skill in skills for skill in service.key_skills):
            fitness += fitness_weights["key_skills"]

    # OPTIONAL
    # Criteria 3: All volunteers must not be assigned more than once per day
    for volunteer_id, assignments in volunteer_assignments.items():
        days_assigned = set()
        for assignment in assignments:
            position = position_dict[assignment.position_id]
            shift = shift_dict[position.shift_id]
            days_assigned.add(shift.time_slot.day)
        if len(days_assigned) == len(assignments):
            fitness += fitness_weights["single_assignment_per_day"]

    # OUTSTANDING
    # Criteria 4: All volunteers must not be assigned more than 4 times total
    for volunteer_id, assignments in volunteer_assignments.items():
        if len(assignments) <= 4:
            fitness += fitness_weights["limited_assignments"]
        else:
            fitness -= fitness_weights["limited_assignments"]

    # OPTIONAL
    # Criteria 5: All assignments for a position concerns volunteers of the same team as much as possible
    for position_id, volunteer_ids in position_volunteers.items():
        teams_assigned = [volunteer_dict[volunteer_id].team for volunteer_id in volunteer_ids]
        if len(set(teams_assigned)) == 1:
            fitness += fitness_weights["team_variety"]

    # OPTIONAL and OUTSTANDING
    # Criteria 6: All assignments respect the volunteer expression regarding their shift preferences, both to be assigned and not to be assigned
    for volunteer_id, assignments in volunteer_assignments.items():
        volunteer = volunteer_dict[volunteer_id]
        for assignment in assignments:
            position = position_dict[assignment.position_id]
            shift_id = position.shift_id
            if shift_id in volunteer.preferred_shifts:
                fitness += fitness_weights["shift_preferences"]
            if shift_id in volunteer.avoid_shifts:
                fitness -= fitness_weights["shift_unavailable"]

    # OUTSTANDING
    # Criteria 7: A volunteer can not be assigned at two different sites at the same time
    for volunteer_id, assignments in volunteer_assignments.items():
        time_slots_assigned = {}
        for assignment in assignments:
            position = position_dict[assignment.position_id]
            shift = shift_dict[position.shift_id]
            day = shift.time_slot.day
            start = shift.time_slot.start
            end = shift.time_slot.end
            site = shift.site
            if (day, start, end) in time_slots_assigned and time_slots_assigned[(day, start, end)] != site:
                fitness -= fitness_weights["no_multiple_sites"]
                break
            time_slots_assigned[(day, start, end)] = site
        else:
            fitness += fitness_weights["no_multiple_sites"]

    # OUTSTANDING
    # Criteria 8: There must not be a vacant position
    for position in positions:
        if len(position_volunteers.get(position.id, [])) < position.min_volunteers:
            fitness -= fitness_weights["no_vacant_positions"]
        else:
            fitness += fitness_weights["no_vacant_positions"]

    return fitness


def selection(population: List[List[Assignment]], volunteers: List[Volunteer], fitness_scores: List[float], tournament_size: int = 3,
              p_select_second_best: float = 0.2) -> List[List[List[Assignment] | List[Volunteer]]]:
    selected = []
    population_size = len(population)

    for _ in range(2):
        # Select `tournament_size` individuals from the population at random
        competitors_idx = random.sample(range(population_size), tournament_size)
        # Sort the indices by fitness score
        sorted_idx = sorted(competitors_idx, key=lambda idx: fitness_scores[idx], reverse=True)

        # Select the best or second-best with a probability of `p_select_second_best`
        if len(sorted_idx) > 1 and random.random() < p_select_second_best:
            selected_idx = sorted_idx[1]
        else:
            selected_idx = sorted_idx[0]

        # Append the selected individual to the selected parents
        selected.append([population[selected_idx], volunteers[selected_idx]])

    return selected


def crossover(parent1: List[Assignment], parent2: List[Assignment]) -> tuple[list[Any], list[Any]]:
    # Cycle Crossover
    parent1_dict = {assignment.position_id: assignment for assignment in parent1}
    parent2_dict = {assignment.position_id: assignment for assignment in parent2}

    positions = set(parent1_dict.keys()).union(set(parent2_dict.keys()))

    # Find cycles
    cycles = []
    visited = set()
    for start_position in positions:
        if start_position not in visited:
            current_position = start_position
            cycle = []
            while True:
                cycle.append(current_position)
                visited.add(current_position)
                # Check if the current position exists in parent1_dict before trying to access it
                if current_position not in parent1_dict:
                    break
                next_position = parent2_dict.get(parent1_dict[current_position].volunteer_id)
                if next_position is None or next_position.position_id == start_position:
                    break
                current_position = next_position.position_id
            cycles.append(cycle)

    # Generate children
    child1_assignments = {}
    child2_assignments = {}
    for i, cycle in enumerate(cycles):
        if i % 2 == 0:
            # For even-index cycles, assign positions from parent1 to child1 and parent2 to child2
            for position_id in cycle:
                if position_id in parent1_dict:
                    child1_assignments[position_id] = parent1_dict[position_id]
                if position_id in parent2_dict:
                    child2_assignments[position_id] = parent2_dict[position_id]
        else:
            # For odd-index cycles, assign positions from parent2 to child1 and parent1 to child2
            for position_id in cycle:
                if position_id in parent2_dict:
                    child1_assignments[position_id] = parent2_dict[position_id]
                if position_id in parent1_dict:
                    child2_assignments[position_id] = parent1_dict[position_id]

    # Convert assignments dictionaries to lists
    child1 = list(child1_assignments.values())
    child2 = list(child2_assignments.values())

    return child1, child2


def mutation(assignments: List[Assignment], positions: List[Position], volunteers: List[Volunteer],
             teams: List[Team], services: List[Service], shifts: List[Shift],
             mutation_rate: float, rare_mutation_rate: float) -> List[Assignment]:
    new_assignments = copy.deepcopy(assignments)

    # Check if mutation should occur
    if random.random() < mutation_rate:

        # Sometimes the child have nothing in common with ancestors, making a one-of-a-kind mutation
        if random.random() < rare_mutation_rate:
            # Generate a single new individual using a modified initialize_population function
            new_individual, _ = map(list, zip(*initialize_population(1, positions, volunteers, teams, services, shifts)))
            return new_individual[0]

        # Select two random indices
        idx1, idx2 = random.sample(range(len(new_assignments)), 2)

        # Get the volunteers assigned to the positions
        volunteer1 = new_assignments[idx1].volunteer_id
        volunteer2 = new_assignments[idx2].volunteer_id

        # Swap the volunteers of the two randomly selected assignments
        new_assignments[idx1] = Assignment(position_id=new_assignments[idx1].position_id, volunteer_id=volunteer2)
        new_assignments[idx2] = Assignment(position_id=new_assignments[idx2].position_id, volunteer_id=volunteer1)

    return new_assignments


def calculate_population_diversity(population):
    # For simplicity, let's define diversity as the average pairwise hamming distance
    total_distance = 0
    num_pairs = 0
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            total_distance += sum(a != b for a, b in zip(population[i], population[j]))
            num_pairs += 1
    return total_distance / num_pairs if num_pairs > 0 else 0


def calculate_fitness_population_diversity(fitness_scores: List[float]) -> float:
    return statistics.stdev(fitness_scores)


def calculate_progress_rate(new_best_fitness, prev_best_fitness, num_generations):
    # The rate of progress in terms of improvement of the best fitness
    return (new_best_fitness - prev_best_fitness) / num_generations


def genetic_algorithm(input_json: Dict, fitness_function=None) -> List[Dict[str, Any]]:
    # Extract values from input JSON
    population_size = input_json["population_size"]
    number_of_generations = input_json["number_of_generations"]
    crossover_rate = input_json["crossover_rate"]
    safe_crossover_rate = crossover_rate
    mutation_rate = input_json["mutation_rate"]
    safe_mutation_rate = mutation_rate
    tournament_size = input_json["tournament_size"]
    select_second_best = input_json["select_second_best"]
    rare_mutation_rate = input_json["rare_mutation_rate"]
    safe_rare_mutation_rate = rare_mutation_rate
    fitness_weights = input_json["fitness_weights"]

    # Extract and create objects from input JSON
    teams = [Team(**t) for t in input_json["teams"]]
    services = [Service(**s) for s in input_json["services"]]
    shifts = [Shift(id=s["id"], time_slot=TimeSlot(**s["time_slot"]), site=s["site"]) for s in input_json["shifts"]]
    positions = [Position(id=p["id"], shift_id=p["shift_id"], service_id=p["service_id"],
                          min_volunteers=p["volunteers"]["min"], max_volunteers=p["volunteers"]["max"],
                          recommended_volunteers=p["volunteers"]["recommended"]) for p in input_json["positions"]]

    # Create dictionaries for easier lookup
    position_dict = {position.id: position for position in positions}
    shift_dict = {shift.id: shift for shift in shifts}
    service_dict = {service.id: service for service in services}
    team_dict = {team.name: team for team in teams}

    # Initialize the population
    population, population_volunteers = map(list, zip(
        *initialize_population(
            population_size=population_size,
            positions=positions,
            teams=teams,
            volunteers=[Volunteer(**v) for v in input_json["volunteers"]],
            services=services,
            shifts=shifts
        )
    ))

    prev_best_fitness = -float('inf')
    prev_diversity = -float('inf')
    progress_rate_threshold = 0.01  # A threshold to detect stagnation
    diversity_threshold = 5  # A threshold to detect decrease in diversity

    # Run the genetic algorithm
    fitness_values = []
    for generation in range(number_of_generations):
        # Calculate fitness
        fitness_values = [
            fitness_function(assignments, positions, volunteers, teams, services, shifts, fitness_weights)
            for assignments, volunteers in zip(population, population_volunteers)
        ]

        # Crossover
        if random.random() < crossover_rate:
            # Selection
            parents, parent_volunteers = zip(
                *selection(population, population_volunteers, fitness_values, tournament_size, select_second_best)
            )

            # Offsprings
            child1, child2 = crossover(*parents)

            # Mutation
            child1 = mutation(
                assignments=child1, positions=positions, volunteers=[Volunteer(**v) for v in input_json["volunteers"]],
                teams=teams, services=services, shifts=shifts,
                mutation_rate=mutation_rate, rare_mutation_rate=rare_mutation_rate
            )
            child2 = mutation(
                assignments=child2, positions=positions, volunteers=[Volunteer(**v) for v in input_json["volunteers"]],
                teams=teams, services=services, shifts=shifts,
                mutation_rate=mutation_rate, rare_mutation_rate=rare_mutation_rate
            )

            # Calculate fitness for the children
            child1_fitness = fitness_function(
                child1, positions, random.choice(parent_volunteers), teams, services, shifts, fitness_weights
            )

            child2_fitness = fitness_function(
                child2, positions, random.choice(parent_volunteers), teams, services, shifts, fitness_weights
            )

            # Replace the two worst individuals with the new children
            min_fitness_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])[:2]

            population[min_fitness_indices[0]] = child1
            fitness_values[min_fitness_indices[0]] = child1_fitness

            population[min_fitness_indices[1]] = child2
            fitness_values[min_fitness_indices[1]] = child2_fitness

        # Every ~10% of the total pool, readjust
        if generation % (number_of_generations / random.randint(7, 12)) == 0:
            # Calculate population diversity
            current_diversity = calculate_fitness_population_diversity(
                fitness_values)  # diversity = calculate_population_diversity(population)

            # Calculate progress rate
            current_best_fitness = max(fitness_values)
            progress_rate = calculate_progress_rate(current_best_fitness, prev_best_fitness, number_of_generations)

            # Dynamic parameter adjustment

            # Increase mutation rate if progress rate is too low or diversity is too low
            if progress_rate < progress_rate_threshold or abs(prev_diversity - current_diversity) < diversity_threshold:
                mutation_rate = min(mutation_rate * 1.2, 1)  # Increase by 20% up to 1.0
                crossover_rate = min(crossover_rate * 1.1, 1)  # Increase by 20% up to 1.0
                rare_mutation_rate = min(rare_mutation_rate * 1.05, 1)  # Increase by 20% up to 1.0
                print(
                    f"Increasing mutation rate to [{mutation_rate},{rare_mutation_rate}], crossover rate to {crossover_rate}")

            # Optionally decrease mutation rate if progress rate is high
            elif progress_rate > progress_rate_threshold:
                mutation_rate = max(mutation_rate * 0.9,
                                    safe_mutation_rate)  # Decrease by 10% down to initial parameter
                crossover_rate = max(crossover_rate * 0.9,
                                     safe_crossover_rate)  # Decrease by 10% down to initial parameter
                rare_mutation_rate = max(rare_mutation_rate * 0.9,
                                         safe_rare_mutation_rate)  # Decrease by 10% down to initial parameter
                print(
                    f"Decreasing mutation rate to [{mutation_rate},{rare_mutation_rate}], crossover rate to {crossover_rate}")

            # Store the current diversity and best fitness for the next iteration
            prev_best_fitness = current_best_fitness
            prev_diversity = current_diversity

        # Optionally print report
        print(
            f"Generation {generation}: Best Fitness = {max(fitness_values)}, Worst Fitness = {min(fitness_values)}, Avg Fitness = {statistics.mean(fitness_values)}, Median Fitness = {statistics.median(fitness_values)}")

    # Find the index of the best solution in the population
    best_solution_index = fitness_values.index(max(fitness_values))
    # Extract the best solution (list of Assignments) and export the generation data
    return report_generation(
        population[best_solution_index],
        position_dict,
        shift_dict,
        service_dict,
        {volunteer.id: volunteer for volunteer in population_volunteers[best_solution_index]},
        team_dict
    )
