import csv
from typing import List, Dict, Any

from algorithm.models import Assignment, Position, Shift, Service, Volunteer, Team


def report_generation(population: List[List[Assignment]], fitness_values: List[int],
                      positions: Dict[str, Position], shifts: Dict[str, Shift],
                      services: Dict[str, Service], volunteers: Dict[str, Volunteer],
                      teams: Dict[str, Team]) -> List[Dict[str, Any]]:
    # Find the index of the best solution in the population
    best_solution_index = fitness_values.index(max(fitness_values))

    # Extract the best solution (list of Assignments)
    best_solution = population[best_solution_index]

    report = []

    # Write a CSV report with these informations for the volunteers
    # volunteer_name, team, team_leader, day, site, time_slot, service, relevant skills
    with open('output/volunteers_report.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["volunteer_name", "team", "team_leader", "day", "site", "time_slot", "service", "relevant_skills"])
        for assignment in best_solution:
            position = positions[assignment.position_id]
            volunteer = volunteers[assignment.volunteer_id]
            shift = shifts[position.shift_id]
            service = services[position.service_id]
            team = teams[volunteer.team]
            day = shift.time_slot.day
            time_slot = f"{shift.time_slot.start} - {shift.time_slot.end}"
            site = shift.site
            relevant_skills = ", ".join([skill for skill in service.key_skills if skill in volunteer.skills])

            writer.writerow(
                [volunteer.id, volunteer.team, team.leader, day, site, time_slot, service.name, relevant_skills])

    # Write a CSV report with these informations for the site managers
    # site, service, day, time_slot, unmet skills, missing volunteers to meet expectations
    with open('output/site_managers_report.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["site", "service", "day", "time_slot", "unmet_skills", "missing_volunteers_to_meet_expectations"])
        position_counts = {}
        for assignment in best_solution:
            position = positions[assignment.position_id]
            position_counts[position.id] = position_counts.get(position.id, 0) + 1

        for position_id, position in positions.items():
            shift = shifts[position.shift_id]
            service = services[position.service_id]
            day = shift.time_slot.day
            time_slot = f"{shift.time_slot.start} - {shift.time_slot.end}"
            site = shift.site

            assigned_volunteers = position_counts.get(position_id, 0)
            missing_volunteers = max(0, position.recommended_volunteers - assigned_volunteers)

            # Calculate unmet skills
            all_skills = [volunteers[assignment.volunteer_id].skills for assignment in best_solution if
                          assignment.position_id == position_id]
            unmet_skills = set(service.key_skills)
            for skills in all_skills:
                unmet_skills.difference_update(set(skills.keys()))
            unmet_skills = ", ".join(unmet_skills)

            writer.writerow([site, service.name, day, time_slot, unmet_skills, missing_volunteers])

    # Write a CSV report with these informations for the event organizers
    # volunteer_name, max assignments, [assignments day1, assignment day2, assignment day...], missed shift preferences
    with open('output/event_organizers_report.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["volunteer_name", "max_assignments", "assignments_days", "missed_shift_preferences"])
        volunteer_assignments = {}
        for assignment in best_solution:
            volunteer_id = assignment.volunteer_id
            if volunteer_id not in volunteer_assignments:
                volunteer_assignments[volunteer_id] = []
            volunteer_assignments[volunteer_id].append(assignment)

        for volunteer_id, assignments in volunteer_assignments.items():
            volunteer = volunteers[volunteer_id]
            max_assignments = len(assignments)
            assignments_days = [shifts[positions[assignment.position_id].shift_id].time_slot.day for assignment in
                                assignments]
            assignments_days = ", ".join(assignments_days)

            missed_shifts = set(volunteer.preferred_shifts) - set(
                [positions[assignment.position_id].shift_id for assignment in assignments])
            missed_shift_preferences = [
                f"{shifts[shift_id].time_slot.day}: {shifts[shift_id].time_slot.start}-{shifts[shift_id].time_slot.end}"
                for shift_id in missed_shifts]
            missed_shift_preferences = ", ".join(missed_shift_preferences)

            writer.writerow([volunteer.id, max_assignments, assignments_days, missed_shift_preferences])

    # Write a CSV report with these informations for the volunteers without assignments
    # volunteer_name
    with open('output/unassigned_volunteers.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["volunteer_name"])

        # Track all volunteer ids in assignments
        assigned_volunteer_ids = set(assignment.volunteer_id for assignment in best_solution)

        # Loop through all volunteers to find those who are not assigned
        for volunteer_id, volunteer in volunteers.items():
            if volunteer_id not in assigned_volunteer_ids:
                writer.writerow([volunteer_id])

    return report
