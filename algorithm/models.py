from typing import List, Dict


class TimeSlot:
    def __init__(self, day: str, start: str, end: str):
        self.day = day
        self.start = start
        self.end = end


class Shift:
    def __init__(self, id: str, time_slot: TimeSlot, site: str):
        self.id = id
        self.time_slot = time_slot
        self.site = site


class Service:
    def __init__(self, id: str, name: str, key_skills: List[str]):
        self.id = id
        self.name = name
        self.key_skills = key_skills


class Team:
    def __init__(self, name: str, leader: str):
        self.name = name
        self.leader = leader


class Volunteer:
    def __init__(self, id: str, skills: Dict[str, int], team: str, preferred_shifts: List[str],
                 avoid_shifts: List[str]):
        self.id = id
        self.skills = skills
        self.team = team
        self.preferred_shifts = preferred_shifts
        self.avoid_shifts = avoid_shifts


class Position:
    def __init__(self, id: str, shift_id: str, service_id: str, min_volunteers: int, max_volunteers: int,
                 recommended_volunteers: int):
        self.id = id
        self.shift_id = shift_id
        self.service_id = service_id
        self.min_volunteers = min_volunteers
        self.max_volunteers = max_volunteers
        self.recommended_volunteers = recommended_volunteers


class Assignment:
    def __init__(self, position_id: str, volunteer_id: str):
        self.position_id = position_id
        self.volunteer_id = volunteer_id
