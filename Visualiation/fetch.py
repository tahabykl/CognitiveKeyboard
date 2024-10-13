import requests
import json



def getJson():
    # Step 1: Define the API endpoint
    url = "https://3fca-67-134-204-47.ngrok-free.app/"

    start_date = "2024-10-12/"
    end_date = "2024-10-12"
    url = f"{url}/all-data/{start_date}{end_date}"
    # Step 2: Send a GET request
    response = requests.get(url)

    # Step 3: Check the response
    if response.status_code == 200:
        # print("Response:", response.text)
        data = response.json()
        ages = [entry['age'] for entry in data]
        races = [entry['race'] for entry in data]
        genders = [entry['gender'] for entry in data]
        # contacts = [entr]
        names = [entry['name'] for entry in data]
        uuids = [entry['uuid'] for entry in data]
        #
    


        # print("Fetched all data successfully:", data)
        print(json.dumps(data, indent=4))
        print("ages: ", ages)
        print("races: ", races)
        print("genders: ", genders)
        print("names: ", names)
        print("uuids: ", uuids)

    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        print("Error message:", response.text)
    return response.json()


getJson()
