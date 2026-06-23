"""
live_post_request.py

Sends a single POST request to the live, deployed API and prints
the status code and response body. Run this manually after deploy
to confirm the live service is actually reachable and returning
correct predictions -- not just that CI passed locally.

Usage:
    python live_post_request.py https://your-app.onrender.com
"""
import sys
import requests

DEFAULT_PAYLOAD = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python live_post_request.py <base_url>")
        print(
            "Example: python live_post_request.py "
            "https://census-income-api.onrender.com"
        )
        sys.exit(1)

    base_url = sys.argv[1].rstrip("/")
    url = f"{base_url}/predict"

    response = requests.post(url, json=DEFAULT_PAYLOAD, timeout=30)

    print(f"URL: {url}")
    print(f"Status code: {response.status_code}")
    print(f"Response body: {response.text}")


if __name__ == "__main__":
    main()
