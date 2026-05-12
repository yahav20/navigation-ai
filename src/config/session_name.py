"""Random session-name generator (e.g. 'happy-traveler', 'cosmic-cartographer')."""
import random

_ADJECTIVES = (
    "happy", "wandering", "curious", "bold", "sleepy", "lucky", "clever", "gentle",
    "brave", "sunny", "cosmic", "merry", "dreamy", "swift", "quiet", "wild",
    "jolly", "rustic", "breezy", "mellow", "spicy", "humble", "noble", "witty",
)
_NOUNS = (
    "traveler", "marco-polo", "voyager", "nomad", "explorer", "pilgrim", "drifter",
    "wayfarer", "compass", "lantern", "atlas", "sailor", "trekker", "rover",
    "scout", "globetrotter", "cartographer", "mariner", "pathfinder", "sherpa",
    "caravan", "wanderer", "ranger", "skipper",
)


def generate_session_name() -> str:
    suffix = random.randint(1000, 9999)
    return f"{random.choice(_ADJECTIVES)}-{random.choice(_NOUNS)}-{suffix}"
