import argparse
import json
from collections import defaultdict
from itertools import chain, combinations, product
from pathlib import Path
from typing import Dict, Optional, Set

from pulp import (
    LpAffineExpression,
    LpBinary,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    getSolver,
    lpSum,
    value,
)

from .util import SchedulingInput

_DUMMY_PARTICIPANT_PREFIX = "DUMMY_PARTICIPANT"


def process_pref_score(preference_score: int, required: bool, mu: float) -> float:
    if required or preference_score == -1:
        return 0
    elif preference_score in [0, 1]:
        return preference_score
    elif preference_score == 2:
        return mu
    else:
        raise NotImplementedError(preference_score)


## TODO könnte auch mit DUMMY_PARTICIPANT_{uuid.uuid4()}_{ak_id} für sichere eindeutigkeit gemacht werden
def get_dummy_participant_id(ak_id: str, dummy_prefix: Optional[str] = None) -> str:
    if not dummy_prefix:
        dummy_prefix = _DUMMY_PARTICIPANT_PREFIX
    return f"{dummy_prefix}_{ak_id}"


def is_participant_dummy(
    participant_id: str, dummy_prefix: Optional[str] = None
) -> bool:
    if not dummy_prefix:
        dummy_prefix = _DUMMY_PARTICIPANT_PREFIX
    return participant_id.startswith(dummy_prefix)


def _construct_constraint_name(name: str, *args) -> str:
    return name + "_" + "_".join(args)


def _set_decision_variable(
    dec_vars: Dict[str, Dict[str, Dict[str, Dict[str, LpVariable]]]],
    ak_id: str,
    timeslot_id: str,
    room_id: str,
    participant_id: str,
    value: float,
    name: Optional[str] = None,
) -> None:
    """Force a decision variable to be a fixed value."""
    if name is not None:
        name = _construct_constraint_name(
            name, ak_id, timeslot_id, room_id, participant_id
        )
    dec_vars[ak_id][timeslot_id][room_id][participant_id].setInitialValue(value)
    dec_vars[ak_id][timeslot_id][room_id][participant_id].fixValue()




def create_lp(
    input_data: SchedulingInput,
    mu: float,
    output_file: str | None = "koma-plan.lp",
) -> None:
    """Create the MILP problem as pulp object.

    Creates the problem with all constraints, preferences and the objective function.

    For a specification of the input JSON format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    For a specification of the MILP, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/LP-formulation

    The MILP models each person to have three kinds of prefences for an AK:
    0 (no preference), 1 (weak preference) and `mu` (strong preference).
    The choice of `mu` is an hyperparameter of the MILP that weights the
    balance between weak and strong preferences.

    Args:
        input_data (SchedulingInput): The input data used to construct the MILP.
        mu (float): The weight associated with a strong preference for an AK.
        output_file (str, optional): If not None, the created LP is written
            as an `.lp` file to this location. Defaults to `koma-plan.lp`.
    """
    # Get values needed from the input_dict
    room_capacities = {room.id: room.capacity for room in input_data.rooms}
    ak_durations = {ak.id: ak.duration for ak in input_data.aks}

    # dict of real participants only (without dummy participants) with their preferences dicts
    real_preferences_dict = {
        participant.id: participant.preferences
        for participant in input_data.participants
    }

    # dict of real participants only (without dummy participants) with numerical preferences
    weighted_preference_dict = {
        participant.id: {
            pref.ak_id: process_pref_score(
                pref.preference_score,
                pref.required,
                mu=mu,
            )
            for pref in participant.preferences
        }
        for participant in input_data.participants
    }

    # Get constraints from input_dict
    participant_time_constraint_dict = {
        participant.id: set(participant.time_constraints)
        for participant in input_data.participants
    }

    participant_room_constraint_dict = {
        participant.id: set(participant.room_constraints)
        for participant in input_data.participants
    }

    ak_time_constraint_dict = {ak.id: set(ak.time_constraints) for ak in input_data.aks}
    ak_room_constraint_dict = {ak.id: set(ak.room_constraints) for ak in input_data.aks}

    room_time_constraint_dict = {
        room.id: set(room.time_constraints) for room in input_data.rooms
    }
    fulfilled_time_constraints = {
        timeslot.id: set(timeslot.fulfilled_time_constraints)
        for block in input_data.timeslot_blocks
        for timeslot in block
    }
    fulfilled_room_constraints = {
        room.id: set(room.fulfilled_room_constraints) for room in input_data.rooms
    }

    def _retrieve_ids(input_iterable) -> Set:
        return {obj.id for obj in input_iterable}

    ak_ids = _retrieve_ids(input_data.aks)
    room_ids = _retrieve_ids(input_data.rooms)
    timeslot_ids = _retrieve_ids(chain.from_iterable(input_data.timeslot_blocks))
    participant_ids = _retrieve_ids(input_data.participants)
    participant_ids = (
        participant_ids.union(  # contains all participants ids (incl. dummy ids)
            {get_dummy_participant_id(ak_id) for ak_id in ak_ids}
        )
    )

    timeslot_block_ids = {
        timeslot.id: (block_idx, timeslot_idx)
        for block_idx, block in enumerate(input_data.timeslot_blocks)
        for timeslot_idx, timeslot in enumerate(block)
    }

    # Create problem
    prob = LpProblem("MLPKoMa", sense=LpMinimize)

    # Create decision variables
    dec_vars = LpVariable.dicts(
        "DecVar", (ak_ids, timeslot_ids, room_ids, participant_ids), cat=LpBinary
    )

    # Set objective function
    #   ∑ᴬ⋅ᵀ⋅ᴿ⋅ᴾ -Pᴬ⋅ᴾ / (Sᴬ ∑_{Pᴬ⋅ᴾ≠0} 1) Yᴬ⋅ᵀ⋅ᴿ⋅ᴾ
    cost_func = LpAffineExpression()
    for participant_id, preferences in real_preferences_dict.items():
        normalizing_factor = len(preferences)
        if normalizing_factor == 0:
            continue
        for ak_id in ak_ids:
            coeff = -weighted_preference_dict[participant_id].get(ak_id, 0)
            coeff /= ak_durations[ak_id] * normalizing_factor
            affine_constraint = lpSum(
                [
                    dec_vars[ak_id][timeslot_id][room_id][participant_id]
                    for timeslot_id, room_id in product(timeslot_ids, room_ids)
                ]
            )
            cost_func += coeff * affine_constraint

    prob += cost_func, "cost_function"

    # Add constraints

    # E1: MaxOneAKperPersonAndTime
    #   ∀ T,P≠Pᴬ: ∑ᴬ⋅ᵀ Yᴬ⋅ᵀ⋅ᴿ⋅ᴾ ≤ 1
    for timeslot_id, participant_id in product(
        timeslot_ids, real_preferences_dict.keys()
    ):
        affine_constraint = lpSum(
            [
                dec_vars[ak_id][timeslot_id][room_id][participant_id]
                for ak_id, room_id in product(ak_ids, room_ids)
            ]
        )
        prob += affine_constraint <= 1, _construct_constraint_name(
            "MaxOneAKperPersonAndTime", timeslot_id, participant_id
        )

    # E2: AKLength
    #   ∀ A: ∑ᵀ⋅ᴿ Yᴬ⋅ᵀ⋅ᴿ⋅ᴾᴬ = Sᴬ
    for ak_id in ak_ids:
        affine_constraint = lpSum(
            [
                dec_vars[ak_id][timeslot_id][room_id][get_dummy_participant_id(ak_id)]
                for timeslot_id, room_id in product(timeslot_ids, room_ids)
            ]
        )
        prob += affine_constraint == ak_durations[ak_id], _construct_constraint_name(
            "AKLength", ak_id
        )

    ## TODO FIXME BUG: Muss =1 oder =0 sein!
    # E3: NoPartialParticipation
    #   ∀ A,P≠Pᴬ: 1/Sᴬ ∑ᵀ⋅ᴿ Yᴬ⋅ᵀ⋅ᴿ⋅ᴾ ≤ 1
    # Z2: PersonNeededForAK
    #   ∀ A,P≠Pᴬ if P essential for A: ∑ᵀ⋅ᴿ Yᴬ⋅ᵀ⋅ᴿ⋅ᴾ = Sᴬ
    for ak_id in ak_ids:
        for participant_id, preferences in real_preferences_dict.items():
            affine_constraint = lpSum(
                [
                    dec_vars[ak_id][timeslot_id][room_id][participant_id]
                    for timeslot_id, room_id in product(timeslot_ids, room_ids)
                ]
            )
            for pref in preferences:
                if pref.ak_id == ak_id:
                    # participant is essential for ak -> set constraint for "PersonNeededForAK"
                    if pref.required:
                        prob += (
                            affine_constraint == ak_durations[ak_id],
                            _construct_constraint_name(
                                "PersonNeededForAK",
                                ak_id,
                                participant_id,
                            ),
                        )  ## TODO Check for fixed value
                    else:  # participant is not essential -> set constraint for "NoPartialParticipation"
                        affine_constraint *= 1 / ak_durations[ak_id]
                        prob += affine_constraint <= 1, _construct_constraint_name(
                            "NoPartialParticipation",
                            ak_id,
                            participant_id,
                        )
                    break
            else:  # participant is not essential -> set constraint for "NoPartialParticipation"
                affine_constraint *= 1 / ak_durations[ak_id]
                prob += affine_constraint <= 1, _construct_constraint_name(
                    "NoPartialParticipation",
                    ak_id,
                    participant_id,
                )

    ## TODO FIXME BUG: Muss =1 oder =0 sein!
    # E4: FixedAKRooms
    #   ∀ A,R: 1 / Sᴬ ∑ᵀ Yᴬ⋅ᵀ⋅ᴿ⋅ᴾᴬ ≤ 1
    for ak_id, room_id in product(ak_ids, room_ids):
        affine_constraint = lpSum(
            [
                dec_vars[ak_id][timeslot_id][room_id][get_dummy_participant_id(ak_id)]
                for timeslot_id in timeslot_ids
            ]
        )
        affine_constraint *= 1 / ak_durations[ak_id]
        prob += affine_constraint <= 1, _construct_constraint_name(
            "FixedAKRooms", ak_id, room_id
        )

    # E5: AKConsecutive
    #   ∀ A,R,Tᵃᵇ,Tᶜᵈ s.t. (a≠c ∨ |b-d|≥Sᴬ): Yᴬ⋅ᵀᵃᵇ⋅ᴿ⋅ᴾᴬ + Yᴬ⋅ᵀᶜᵈ⋅ᴿ⋅ᴾᴬ ≤ 1
    for ak_id, room_id in product(ak_ids, room_ids):
        for timeslot_id_a, timeslot_id_b in combinations(timeslot_ids, 2):
            block_idx_a, slot_in_block_idx_a = timeslot_block_ids[timeslot_id_a]
            block_idx_b, slot_in_block_idx_b = timeslot_block_ids[timeslot_id_b]
            if (
                block_idx_a != block_idx_b
                or abs(slot_in_block_idx_a - slot_in_block_idx_b) >= ak_durations[ak_id]
            ):  # if two timeslots are too far apart to be consecutive
                affine_constraint = lpSum(
                    [
                        dec_vars[ak_id][timeslot_id_a][room_id][
                            get_dummy_participant_id(ak_id)
                        ],
                        dec_vars[ak_id][timeslot_id_b][room_id][
                            get_dummy_participant_id(ak_id)
                        ],
                    ]
                )
                prob += (
                    affine_constraint <= 1,
                    _construct_constraint_name(  # forbid the ak to happen in both of them
                        "AKConsecutive", ak_id, room_id, timeslot_id_a, timeslot_id_b
                    ),
                )

    # E6: PersonVisitingAKAtRightTimeAndRoom
    #   ∀ A,T,R,P≠Pᴬ: Yᴬ⋅ᵀ⋅ᴿ⋅ᴾᴬ - Yᴬ⋅ᵀ⋅ᴿ⋅ᴾ ≥ 0
    for ak_id, timeslot_id, room_id, participant_id in product(
        ak_ids, timeslot_ids, room_ids, real_preferences_dict.keys()
    ):
        affine_constraint = LpAffineExpression(
            dec_vars[ak_id][timeslot_id][room_id][get_dummy_participant_id(ak_id)]
        )
        affine_constraint -= LpAffineExpression(
            dec_vars[ak_id][timeslot_id][room_id][participant_id]
        )
        prob += affine_constraint >= 0, _construct_constraint_name(
            "PersonVisitingAKAtRightTimeAndRoom",
            ak_id,
            timeslot_id,
            room_id,
            participant_id,
        )

    # E7: Roomsizes
    #   ∀ R,T: ∑_{A, P≠Pᴬ} Yᴬ⋅ᵀ⋅ᴿ⋅ᴾ ≤ Kᴿ
    for room_id, timeslot_id in product(room_ids, timeslot_ids):
        affine_constraint = lpSum(
            [
                dec_vars[ak_id][timeslot_id][room_id][participant_id]
                for ak_id, participant_id in product(ak_ids, real_preferences_dict)
            ]
        )
        prob += affine_constraint <= room_capacities[
            room_id
        ], _construct_constraint_name("Roomsizes", room_id, timeslot_id)

    # E8: DummyPersonOneAk
    #   ∀ T,R,B≠A: Yᴮ⋅ᵀ⋅ᴿ⋅ᴾᴬ = 0
    for timeslot_id, room_id, ak_id, dummy_ak_id in product(
        timeslot_ids, room_ids, ak_ids, ak_ids
    ):
        if ak_id == dummy_ak_id:
            continue
        _set_decision_variable(
            dec_vars,
            ak_id,
            timeslot_id,
            room_id,
            get_dummy_participant_id(dummy_ak_id),
            value=0,
            name="DummyPersonOneAk",
        )

    # Z1: PersonNotInterestedInAK
    #   ∀ A,T,R,P: If Pᴾ⋅ᴬ=0: Yᴬ⋅ᵀ⋅ᴿ⋅ᴾ = 0 (non-dummy P)
    for participant_id, preferences in real_preferences_dict.items():
        pref_aks = {
            pref.ak_id for pref in preferences
        }  # aks not in pref_aks have Pᴾ⋅ᴬ=0 implicitly
        for ak_id, timeslot_id, room_id in product(
            ak_ids.difference(pref_aks), timeslot_ids, room_ids
        ):
            _set_decision_variable(
                dec_vars,
                ak_id,
                timeslot_id,
                room_id,
                participant_id,
                value=0,
                name="PersonNotInterestedInAK",
            )

    for participant_id in real_preferences_dict:
        # Z3: TimeImpossibleForPerson (real person P cannot attend AKs with timeslot T)
        #   ∀ A,T,R,P: If P cannot attend at T: Yᴬ⋅ᵀ⋅ᴿ⋅ᴾ=0
        for timeslot_id in timeslot_ids:
            if participant_time_constraint_dict[participant_id].difference(
                fulfilled_time_constraints[timeslot_id]
            ):
                for ak_id, room_id in product(ak_ids, room_ids):
                    _set_decision_variable(
                        dec_vars,
                        ak_id,
                        timeslot_id,
                        room_id,
                        participant_id,
                        value=0,
                        name="TimeImpossibleForPerson",
                    )

        # Z4: RoomImpossibleForPerson (Real person P cannot attend AKs with room R)
        #   ∀ A,T,R,P: If P cannot attend in R: Yᴬ⋅ᵀ⋅ᴿ⋅ᴾ=0
        for room_id in room_ids:
            if participant_room_constraint_dict[participant_id].difference(
                fulfilled_room_constraints[room_id]
            ):
                for ak_id, timeslot_id in product(ak_ids, timeslot_ids):
                    _set_decision_variable(
                        dec_vars,
                        ak_id,
                        timeslot_id,
                        room_id,
                        participant_id,
                        value=0,
                        name="RoomImpossibleForPerson",
                    )

    for ak_id in ak_ids:
        # Z5: TimeImpossibleForAK
        #   ∀ A,T,R,P: If A cannot happen in timeslot T: Yᴬ⋅ᵀ⋅ᴿ⋅ᴾ=0
        for timeslot_id in timeslot_ids:
            if ak_time_constraint_dict[ak_id].difference(
                fulfilled_time_constraints[timeslot_id]
            ):
                for participant_id, room_id in product(participant_ids, room_ids):
                    _set_decision_variable(
                        dec_vars,
                        ak_id,
                        timeslot_id,
                        room_id,
                        participant_id,
                        value=0,
                        name="TimeImpossibleForAK",
                    )
        # Z6: RoomImpossibleForAK
        #   ∀ A,T,R,P: If A cannot happen in room R: Yᴬ⋅ᵀ⋅ᴿ⋅ᴾ=0
        for room_id in room_ids:
            if ak_room_constraint_dict[ak_id].difference(
                fulfilled_room_constraints[room_id]
            ):
                for participant_id, timeslot_id in product(
                    participant_ids, timeslot_ids
                ):
                    _set_decision_variable(
                        dec_vars,
                        ak_id,
                        timeslot_id,
                        room_id,
                        participant_id,
                        value=0,
                        name="RoomImpossibleForAK",
                    )

    # Z7: TimeImpossibleForRoom
    #   ∀ A,T,R,P: If room R is not available in timeslot T: Yᴬ⋅ᵀ⋅ᴿ⋅ᴾ=0
    for room_id, timeslot_id in product(room_ids, timeslot_ids):
        if room_time_constraint_dict[room_id].difference(
            fulfilled_time_constraints[timeslot_id]
        ):
            for participant_id, ak_id in product(participant_ids, ak_ids):
                _set_decision_variable(
                    dec_vars,
                    ak_id,
                    timeslot_id,
                    room_id,
                    participant_id,
                    value=0,
                    name="TimeImpossibleForRoom",
                )

    # Z8: NoAKCollision
    #   ∀ T, AKs A,B with A and B may not overlap: ∑ᴿ Yᴬ⋅ᵀ⋅ᴿ⋅ᴾᴬ + Yᴮ⋅ᵀ⋅ᴿ⋅ᴾᴮ ≤ 1
    ## TODO: Not implemented yet

    # The problem data is written to an .lp file
    if output_file is not None:
        prob.writeLP(output_file)

    return prob

def solve_scheduling(
    input_data: SchedulingInput,
    mu: float,
    solver_name: str | None = None,
    output_file: str | None = "koma-plan.lp",
    **solver_kwargs,
) -> None:
    """Solve the scheduling problem.

    Solves the MILP scheduling problem described by the input data using an MILP
    formulation.

    For a specification of the input format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    For a specification of the MILP used, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/LP-formulation

    The MILP models each person to have three kinds of prefences for an AK:
    0 (no preference), 1 (weak preference) and `mu` (strong preference).
    The choice of `mu` is an hyperparameter of the MILP that weights the
    balance between weak and strong preferences.

    Args:
        input_data (SchedulingInput): The input data used to construct the MILP.
        mu (float): The weight associated with a strong preference for an AK.
        output_file (str, optional): If not None, the created LP is written
            as an `.lp` file to this location. Defaults to `koma-plan.lp`.
        solver_name (str, optional): The solver to use. If None, uses pulp's
            default solver. Defaults to None.
        **solver_kwargs: kwargs are passed to the solver.
    """
    prob = create_lp(input_data, mu)

    if solver_name:
        solver = getSolver(solver_name, **kwargs_dict)
    else:
        solver = None
    # The problem is solved using PuLP's choice of Solver
    res = prob.solve(solver)

    # The status of the solution is printed to the screen
    print("Status:", LpStatus[prob.status])

    tmp_res_dir = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for ak_id, timeslot_id, room_id, participant_id in product(
        ak_ids, timeslot_ids, room_ids, real_preferences_dict.keys()
    ):
        if value(dec_vars[ak_id][timeslot_id][room_id][participant_id]) == 1:
            tmp_res_dir[ak_id][room_id]["timeslot_ids"].add(timeslot_id)
            tmp_res_dir[ak_id][room_id]["participant_ids"].add(participant_id)

    output_dict = {}
    output_dict["scheduled_aks"] = [
        {
            "ak_id": ak_id,
            "room_id": room_id,
            "timeslot_ids": list(subsubdict["timeslot_ids"]),
            "participant_ids": list(subsubdict["participant_ids"]),
        }
        for ak_id, subdict in tmp_res_dir.items()
        for room_id, subsubdict in subdict.items()
    ]
    output_dict["input"] = input_data.to_dict()

    with open("output.json", "w") as output_file:
        json.dump(output_dict, output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=float, default=2)
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument("--solver-path", type=str)
    parser.add_argument("--warm-start", action="store_true", default=False)
    parser.add_argument(
        "--timelimit",
        type=float,
        default=None,
        help="Timelimit as stopping criterion (in seconds)",
    )
    parser.add_argument(
        "--gap_rel", type=float, default=None, help="Relative gap as stopping criterion"
    )
    parser.add_argument(
        "--gap_abs", type=float, default=None, help="Absolute gap as stopping criterion"
    )
    parser.add_argument(
        "--threads", type=int, default=None, help="Number of threads to use"
    )
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    solver_kwargs = {}
    if args.solver_path:
        solver_kwargs["path"] = args.solver_path
    if args.warm_start:
        solver_kwargs["warmStart"] = True
    if args.timelimit:
        solver_kwargs["timeLimit"] = args.timelimit
    if args.gap_rel:
        solver_kwargs["gapRel"] = args.gap_rel
    if args.gap_abs:
        solver_kwargs["gapAbs"] = args.gap_abs
    if args.threads:
        solver_kwargs["Threads"] = args.threads

    json_file = Path(args.path)
    assert json_file.suffix == ".json"
    with json_file.open("r") as f:
        input_dict = json.load(f)

    solve_scheduling(
        SchedulingInput.from_dict(input_dict), args.mu, args.solver, **solver_kwargs
    )


if __name__ == "__main__":
    main()
