import random
import csv
import json
import statistics
import random
from typing import List, Dict, Any
from collections import defaultdict

from models import TimeSlot, Shift, Service, Team, Volunteer, Position, Assignment
from run import final_json


def initialize_population(population_size: int, positions: List[Position], volunteers: List[Volunteer],
                          teams: List[Team], services: List[Service], shifts: List[Shift]) -> List[List[Assignment]]:
    population = []

    # Create dictionaries for easier lookup
    volunteer_dict = {volunteer.id: volunteer for volunteer in volunteers}
    position_dict = {position.id: position for position in positions}
    service_dict = {service.id: service for service in services}
    shift_dict = {shift.id: shift for shift in shifts}

    for gen in range(population_size):
        assignments = []
        position_volunteers = defaultdict(set)

        # Count of assignments per volunteer for the current population
        volunteer_assignment_count = defaultdict(int)
        # Count of assignments per day for each volunteer for the current population
        volunteer_daily_assignment_count = defaultdict(lambda: defaultdict(int))

        # Assign volunteers considering the constraints
        for position in positions:
            service = service_dict[position.service_id]
            required_skills = set(service.key_skills)
            shift = shift_dict[position.shift_id]

            candidates = [vol for vol in volunteers if
                          any(skill in vol.skills for skill in required_skills) and
                          volunteer_assignment_count[vol.id] < 4 and
                          volunteer_daily_assignment_count[vol.id][shift.time_slot.day] < 1]

            # Determine the team that has already been assigned to the current position
            assigned_volunteer_ids = position_volunteers[position.id]
            assigned_teams = {volunteer_dict[vol_id].team for vol_id in assigned_volunteer_ids}

            # Prefer volunteers from the same team
            common_team = None
            if assigned_teams:
                common_team = assigned_teams.pop()
            preferred_candidates = [vol for vol in candidates if vol.team == common_team]

            num_volunteers = random.randint(position.min_volunteers, position.max_volunteers)

            # Select volunteers
            selected_volunteers = random.sample(preferred_candidates or candidates,
                                                min(len(candidates), num_volunteers, position.recommended_volunteers))

            for volunteer in selected_volunteers:
                assignments.append(Assignment(position_id=position.id, volunteer_id=volunteer.id))
                position_volunteers[position.id].add(volunteer.id)
                volunteer_assignment_count[volunteer.id] += 1
                volunteer_daily_assignment_count[volunteer.id][shift.time_slot.day] += 1

        # Attempt to add preferred shifts for remaining volunteers
        for volunteer in volunteers:
            if volunteer_assignment_count[volunteer.id] < 4:
                preferred_shifts = set(volunteer.preferred_shifts) - set(
                    position_dict[assignment.position_id].shift_id for assignment in assignments if
                    assignment.volunteer_id == volunteer.id)

                for shift_id in preferred_shifts:
                    available_positions = [position for position in positions if position.shift_id == shift_id and
                                           len(position_volunteers[position.id]) < position.max_volunteers]

                    if available_positions:
                        selected_position = random.choice(available_positions)
                        assignments.append(Assignment(position_id=selected_position.id, volunteer_id=volunteer.id))
                        position_volunteers[selected_position.id].add(volunteer.id)
                        volunteer_assignment_count[volunteer.id] += 1
                        volunteer_daily_assignment_count[volunteer.id][shift_dict[shift_id].time_slot.day] += 1

        population.append(assignments)

        print(f'Generated population {gen} out of total pool {population_size}.')

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

    # Criteria 1: Each position has the recommended number of volunteers at best or between min/max
    for position_id, volunteer_ids in position_volunteers.items():
        volunteer_tot = len(volunteer_ids)
        if position_dict[position_id].min_volunteers <= volunteer_tot <= position_dict[position_id].max_volunteers:
            fitness += fitness_weights["recommended_volunteers"] - abs(
                volunteer_tot - position_dict[position_id].recommended_volunteers)

    # Criteria 2: Each service has at least a volunteer per every key skill required by the service
    for position in positions:
        service = service_dict[position.service_id]
        volunteers_in_position = [volunteer_dict[vol_id] for vol_id in position_volunteers.get(position.id, [])]
        skills = set()
        for v in volunteers_in_position:
            skills.update(v.skills.keys())
        if all(skill in skills for skill in service.key_skills):
            fitness += fitness_weights["key_skills"]

    # Criteria 3: All volunteers must not be assigned more than once per day
    for volunteer_id, assignments in volunteer_assignments.items():
        days_assigned = set()
        for assignment in assignments:
            position = position_dict[assignment.position_id]
            shift = shift_dict[position.shift_id]
            days_assigned.add(shift.time_slot.day)
        if len(days_assigned) == len(assignments):
            fitness += fitness_weights["single_assignment_per_day"]

    # Criteria 4: All volunteers must not be assigned more than 4 times total
    for volunteer_id, assignments in volunteer_assignments.items():
        if len(assignments) <= 4:
            fitness += fitness_weights["limited_assignments"]

    # Criteria 5: All assignments for a position concerns volunteers of the same team as much as possible
    for position_id, volunteer_ids in position_volunteers.items():
        teams_assigned = [volunteer_dict[volunteer_id].team for volunteer_id in volunteer_ids]
        if len(set(teams_assigned)) == 1:
            fitness += fitness_weights["team_variety"]

    # Criteria 6: All assignments respect the volunteer expression regarding their shift preferences, both to be assigned and not to be assigned
    for volunteer_id, assignments in volunteer_assignments.items():
        volunteer = volunteer_dict[volunteer_id]
        for assignment in assignments:
            position = position_dict[assignment.position_id]
            shift_id = position.shift_id
            if shift_id in volunteer.preferred_shifts:
                fitness += fitness_weights["shift_preferences"]
            if shift_id in volunteer.avoid_shifts:
                fitness -= fitness_weights["shift_preferences"]

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

    # Criteria 8: There must not be a vacant position
    for position in positions:
        if len(position_volunteers.get(position.id, [])) < position.min_volunteers:
            fitness -= fitness_weights["no_vacant_positions"]
        else:
            fitness += fitness_weights["no_vacant_positions"]

    return fitness


def calculate_fitness(population: List[List[Assignment]], positions: List[Position], volunteers: List[Volunteer],
                      teams: List[Team], services: List[Service], shifts: List[Shift],
                      fitness_weights: Dict[str, int]) -> List[int]:
    """
    This function calculates the fitness of all solutions in the population.
    """
    return [default_fitness_function(assignments, positions, volunteers, teams, services, shifts, fitness_weights) for
            assignments in population]


def selection(population: List[List[Assignment]], fitness_scores: List[float], tournament_size: int = 3,
              p_select_second_best: float = 0.2) -> List[List[Assignment]]:
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
        selected.append(population[selected_idx])

    return selected


def crossover(parent1: List[Assignment], parent2: List[Assignment]) -> list[Assignment]:
    # Uniform Crossover
    child1 = []
    child2 = []
    max_length = max(len(parent1), len(parent2))
    for i in range(max_length):
        if i < len(parent1) and i < len(parent2):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        elif i < len(parent1):
            child1.append(parent1[i])
        else:
            child2.append(parent2[i])
    if max_length > 0:
        return random.sample(child1 + child2, random.randint(
            min(len(parent1), len(parent2)),
            max(len(parent1), len(parent2)))
                             )
    else:
        return []


def mutation(assignments: List[Assignment], positions: List[Position], volunteers: List[Volunteer],
             mutation_rate: float) -> List[Assignment]:
    new_assignments = assignments[:]

    for i in range(len(new_assignments)):
        if random.random() < mutation_rate:
            # Select a random position and volunteer
            random_position = random.choice(positions)
            random_volunteer = random.choice(volunteers)

            # Mutate the current assignment by changing the position and/or volunteer
            new_assignments[i] = Assignment(position_id=random_position.id, volunteer_id=random_volunteer.id)

    return new_assignments


def report_generation(population: List[List[Assignment]], fitness_values: List[int],
                      positions: Dict[str, Position], shifts: Dict[str, Shift],
                      services: Dict[str, Service], volunteers: Dict[str, Volunteer],
                      teams: Dict[str, Team]) -> List[Dict[str, Any]]:
    # Find the index of the best solution in the population
    best_solution_index = fitness_values.index(max(fitness_values))

    # Extract the best solution (list of Assignments)
    best_solution = population[best_solution_index]

    report = []

    # Populate the report with relevant information
    for assignment in best_solution:
        position = positions[assignment.position_id]
        shift = shifts[position.shift_id]
        service = services[position.service_id]
        volunteer = volunteers[assignment.volunteer_id]
        team = teams[volunteer.team]

        entry = {
            "volunteer_name": volunteer.id,
            "team": volunteer.team,
            "team_leader": team.leader,
            "site": shift.site,
            "time_slot": f"{shift.time_slot.start} - {shift.time_slot.end}",
            "service": service.name
        }
        report.append(entry)

    # Optionally, write the report to a CSV file
    with open('report.csv', 'w', newline='') as csvfile:
        fieldnames = ["volunteer_name", "team", "team_leader", "site", "time_slot", "service"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in report:
            writer.writerow(row)

    return report


def genetic_algorithm(input_json: Dict, custom_fitness_function=None) -> List[Dict[str, Any]]:
    # Extract values from input JSON
    population_size = input_json["population_size"]
    number_of_generations = input_json["number_of_generations"]
    crossover_rate = input_json["crossover_rate"]
    mutation_rate = input_json["mutation_rate"]
    tournament_size = input_json["tournament_size"]
    select_second_best = input_json["select_second_best"]
    fitness_weights = input_json["fitness_weights"]

    # Extract and create objects from input JSON
    volunteers = [Volunteer(**v) for v in input_json["volunteers"]]
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
    volunteer_dict = {volunteer.id: volunteer for volunteer in volunteers}
    team_dict = {team.name: team for team in teams}

    # Initialize the population
    population = initialize_population(
        population_size=population_size,
        positions=positions,
        teams=teams,
        volunteers=volunteers,
        services=services,
        shifts=shifts
    )

    # Run the genetic algorithm
    fitness_values = []
    for generation in range(number_of_generations):
        # Calculate fitness
        if custom_fitness_function:
            fitness_values = [
                custom_fitness_function(assignment, positions, volunteers, teams, services, shifts, fitness_weights)
                for assignment in population]
        else:
            fitness_values = calculate_fitness(population, positions, volunteers, teams, services, shifts,
                                               fitness_weights)

        # Selection
        parents = selection(population, fitness_values, tournament_size, select_second_best)

        # Crossover
        if random.random() < crossover_rate:
            child = crossover(*parents)

            # Mutation
            child = mutation(child, positions, volunteers, mutation_rate)

            # Calculate fitness for the child
            if custom_fitness_function:
                child_fitness = custom_fitness_function(child, positions, volunteers, teams, services, shifts,
                                                        fitness_weights)
            else:
                child_fitness = default_fitness_function(child, positions, volunteers, teams, services, shifts,
                                                         fitness_weights)

            # Replace the worst individual with the new child
            min_fitness_index = fitness_values.index(min(fitness_values))
            population[min_fitness_index] = child
            fitness_values[min_fitness_index] = child_fitness

        # Find the index of the best solution in the population
        # best_solution_index = fitness_values.index(max(fitness_values))
        # Extract the best solution (list of Assignments)
        # best_solution = population[best_solution_index]

        # Find the index of the worst solution in the population
        # worst_solution_index = fitness_values.index(min(fitness_values))
        # Extract the best solution (list of Assignments)
        # worst_solution = population[worst_solution_index]

        # Optionally print report
        print(
            f"Generation {generation}: Best Fitness = {max(fitness_values)}, Worst Fitness = {min(fitness_values)}, Avg Fitness = {statistics.mean(fitness_values)}, Median Fitness = {statistics.median(fitness_values)}")

    # Report generation
    return report_generation(
        population,
        fitness_values,
        position_dict,
        shift_dict,
        service_dict,
        volunteer_dict,
        team_dict
    )


# Print the merged input JSON
print(json.dumps(final_json, indent=4))

# Run the genetic algorithm
result = genetic_algorithm(final_json)

# `result` will be a list of dictionaries with the optimal volunteer assignment schedule
