from tools.tools import fetch_flights

if __name__ == "__main__":
    flights = fetch_flights.invoke({"origin": "London", "destination": "Paris"})
    for flight in flights:
        print(flight)