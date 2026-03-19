from tools import fetch_flights

if __name__ == "__main__":
    flights = fetch_flights("London","Paris")
    for flight in flights:
        print(flight)