import os
import json

TRAVEL_DB = './data/travel_db.json'

def load_travel_db():
    """
    Load the travel database from a JSON file.
    """
    if not os.path.exists(TRAVEL_DB):
        return None
    with open(TRAVEL_DB, 'r') as file:
        return json.load(file)