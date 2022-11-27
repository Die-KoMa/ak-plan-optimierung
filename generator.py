import json
import random
import argparse


def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persons", type=int, default=30)
    parser.add_argument("--aks", type=int, default=10)
    parser.add_argument("--rooms", type=int, default=4)
    args = parser.parse_args()

    # we have one hour time slots
    # on Tuesday we go from 8-18, Wednesday from 8-16, Thursday from 8-16, Friday from 8-18
    num_time_slots_collection = [10, 8, 8, 10]

    possible_time_constraints = [
        "ResoAK",
        "Dienstag",
        "Mittwoch",
        "Donnerstag",
        "Freitag",
    ]
    possible_room_constraints = ["barrierefrei", "Beamer"]

    # create timeslots:
    list_of_time_blocks = []
    # an index that remembers which day it is
    index = 1
    # we need to count the amount of slots that we already created in order to have unique ids
    time_slots_that_already_passed = 0
    for num_time_slots in num_time_slots_collection:
        list_of_time_slots = []
        for time_slot_number in range(num_time_slots):
            fulfilled_time_constraints = [possible_time_constraints[index]]
            if index == 1:
                # die Reso Aks sollen alle am ersten Tag stattfinden
                fulfilled_time_constraints.append("ResoAK")
            # we create the time slot
            time_slot = {
                "id": str(time_slots_that_already_passed + time_slot_number),
                "info": {
                    "start": possible_time_constraints[index]
                    + ", "
                    + str(8 + time_slot_number)
                    + "Uhr"
                },
                "fulfilled_time_constraints": fulfilled_time_constraints,
            }
            # we add the time slot to the list of time slots in this block
            list_of_time_slots.append(time_slot)
        # we add the block to the list of blocks
        list_of_time_blocks.append(list_of_time_slots)
        index += 1
        time_slots_that_already_passed += num_time_slots
    time_slot_dictionary = {
        "info": {"duration": "1 Stunde"},
        "blocks": list_of_time_blocks,
    }

    # create rooms:
    room1 = {
        "id": "1",
        "info": {"name": "Raum1"},
        "capacity": 25,
        "fulfilled_room_constraints": ["barrierefrei"],
        "time_constraints": [],  # ["Dienstag", "Mittwoch", "Donnerstag", "Freitag"],
    }
    room2 = {
        "id": "2",
        "info": {"name": "Raum2"},
        "capacity": 30,
        "fulfilled_room_constraints": ["barrierefrei", "Beamer"],
        "time_constraints": [],  # ["Dienstag", "Mittwoch", "Donnerstag", "Freitag"],
    }
    room3 = {
        "id": "3",
        "info": {"name": "Raum3"},
        "capacity": 20,
        "fulfilled_room_constraints": [],
        "time_constraints": [],  # ["Mittwoch", "Donnerstag", "Freitag"],
    }
    room4 = {
        "id": "4",
        "info": {"name": "Raum4"},
        "capacity": 25,
        "fulfilled_room_constraints": ["Beamer", "barrierefrei"],
        "time_constraints": [],  # ["Dienstag"],
    }

    # create aks
    list_of_aks = []
    for index in range(args.aks):
        # random number 1 or 2 for the duration
        duration = random.choice([1, 2])
        # with probability 10% the Ak needs a Beamer, with probability 90%, the AK doesn't
        room_constraints = (
            []
        )  # random.choices([[], ["Beamer"]], weights=[0.9, 0.1], k=1)[0]
        # with probability 20% the AK is a ResoAK
        is_reso_ak = random.choices([True, False], weights=[0.2, 0.8], k=1)[0]
        time_constraints = []
        # if is_reso_ak:
        #     time_constraints = ["ResoAK"]

        # write down the ak
        ak = {
            "id": str(index),
            "duration": duration,
            "properties": [],
            "room_constraints": room_constraints,
            "time_constraints": time_constraints,
            "info": {
                "name": "Name of an AK",
                "head": "Name of the head of the AK",
                "description": "Short description of the AK",
                "reso": is_reso_ak,
            },
        }
        list_of_aks.append(ak)

    list_of_participants = []
    # create participants
    for index in range(args.persons):
        preferences = []
        # generate preferences for each AK:
        for ak_index in range(args.aks):
            is_required = random.choices([True, False], weights=[0.1, 0.9], k=1)[0]
            if is_required:
                preference_score = -1
            else:
                preference_score = random.choice([0, 1, 2])

            if not preference_score == 0:
                ak_preferences = {
                    "ak_id": str(ak_index),
                    "required": False,
                    "preference_score": preference_score,
                }
                preferences.append(ak_preferences)

        participant = {
            "id": str(index),
            "info": {"name": "Name of the Person"},
            "preferences": preferences,
            "room_constraints": [],
            "time_constraints": [],
        }
        list_of_participants.append(participant)

    # create dictionary that we later write into the json-file
    dictionary = {
        "aks": list_of_aks,
        "rooms": [room1, room2, room3, room4],
        "participants": list_of_participants,
        "timeslots": time_slot_dictionary,
        "info": "DummySet",
    }

    # print(dictionary)
    with open("dummy_set.json", "w") as output_file:
        json.dump(dictionary, output_file)


if __name__ == "__main__":
    generate()
