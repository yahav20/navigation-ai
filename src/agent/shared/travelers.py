"""Shared traveler-count and hotel-room rules.

Single source of truth for how group size maps to hotel rooms, imported by both
the metadata node (new-plan path) and the adjustments node (update path) so the
logic isn't duplicated and stays unit-testable.

Room policy (agreed): a room holds up to 2 adults + 1 child (capacity 3). Rooms
are auto-derived from the group size, EXCEPT when the user explicitly names a
room total or asks to add/remove a room in the same message — an explicit room
instruction always wins over the auto-recompute.
"""
import math

from agent.core.state import AgentState


def compute_default_rooms(adults: int, children: int) -> int:
    """Derive the default hotel-room count for a group (2 adults + 1 child per room)."""
    adults = max(0, adults)
    children = max(0, children)
    if adults + children == 0:
        return 1
    return max(1, math.ceil(adults / 2), math.ceil((adults + children) / 3))


def apply_traveler_updates(
    state: AgentState,
    *,
    new_adults: int | None,
    new_children: int | None,
    new_rooms_abs: int | None,
    rooms_delta: int | None,
) -> dict:
    """Resolve people/room state updates per the agreed semantics.

    Returns only the keys that actually changed. An explicit room instruction
    (absolute total or +/- delta) always wins over the people-driven recompute;
    otherwise any change to the group size recomputes the default room count.
    """
    updates: dict = {}

    # Keep track of whether each field is actually in state vs. just defaulted.
    # This matters for the "1 adult" case: new_adults=1 equals the default (1)
    # but the field is still None in state, so enrichment would ask again.
    state_adults   = state.get("num_adults")    # None = never explicitly set
    state_children = state.get("num_children")  # None = never explicitly set
    cur_adults   = state_adults   if state_adults   is not None else 1
    cur_children = state_children if state_children is not None else 0
    cur_rooms    = state.get("num_rooms", compute_default_rooms(cur_adults, cur_children))

    people_changed = False
    # Update adults if: (a) never set in state yet (even if value == default 1),
    # or (b) value actually changes from what's already stored.
    if new_adults is not None and (state_adults is None or new_adults != cur_adults):
        updates["num_adults"] = max(0, new_adults)
        cur_adults = updates["num_adults"]
        people_changed = True
        # Seed children to 0 the first time adults are explicitly provided so
        # the field is always a concrete int once the group size is known.
        if state_children is None and new_children is None:
            updates["num_children"] = 0
            cur_children = 0
    # Same "never set vs default" logic for children.
    if new_children is not None and (state_children is None or new_children != cur_children):
        updates["num_children"] = max(0, new_children)
        cur_children = updates["num_children"]
        people_changed = True

    # Explicit room instruction always wins over the auto-recompute.
    if new_rooms_abs is not None:
        updates["num_rooms"] = max(1, new_rooms_abs)
    elif rooms_delta:
        updates["num_rooms"] = max(1, cur_rooms + rooms_delta)
    elif people_changed:
        updates["num_rooms"] = compute_default_rooms(cur_adults, cur_children)

    return updates
