import requests
import json


def updateDatabase(parking):
    url = "https://europe-west1-msiosm.cloudfunctions.net/parkingDataV2"

    payload = {
        # This is the parking id
        "parkingId": "RdDgeKUx4bnyWs6kjBs3",

        # this object contains an array of the parkings lots with their state (avialble or occupied)
        "lots": [
            {
                "id": 1,
                "state": "occupied" if parking.slots[0].full else "available"
            },
            {
                "id": 2,
                "state": "occupied" if parking.slots[1].full else "available"
            },
            {
                "id": 3,
                "state": "occupied" if parking.slots[2].full else "available"
            },
            {
                "id": 4,
                "state": "occupied" if parking.slots[3].full else "available"
            },
            {
                "id": 5,
                "state": "occupied" if parking.slots[4].full else "available"
            },
            {
                "id": 6,
                "state": "occupied" if parking.slots[5].full else "available"
            }
        ]
    }

    headers = {
        'Content-Type': "application/json",
    }

    response = requests.request(
        "POST", url, data=json.dumps(payload), headers=headers)

    return response.text

def initDatabase():
    url = "https://europe-west1-msiosm.cloudfunctions.net/parkingDataV2"

    payload = {
        # This is the parking id
        "parkingId": "RdDgeKUx4bnyWs6kjBs3",

        # this object contains an array of the parkings lots with their state (avialble or occupied)
        "lots": [
            {
                "id": 1,
                "state": "available"
            },
            {
                "id": 2,
                "state": "available"
            },
            {
                "id": 3,
                "state": "available"
            },
            {
                "id": 4,
                "state": "available"
            },
            {
                "id": 5,
                "state": "available"
            },
            {
                "id": 6,
                "state": "available"
            }
        ]
    }

    headers = {
        'Content-Type': "application/json",
    }

    response = requests.request(
        "POST", url, data=json.dumps(payload), headers=headers)

    return response.text
# the online dashboard is updated in realtime via Firebase observer pattern.
