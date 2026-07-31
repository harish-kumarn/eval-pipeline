"""
retriever.py
------------
VisionRAG retriever: queries the VisionRAG API and returns structured results.
"""

import random
import requests



def reinitialize_chat(user_id,collection_name):
    """
    Calls the reinit chat API for a given user and document.
    """
    base_url = "http://localhost:8000/api/v1"
    url = f"{base_url}/users/{user_id}/chat/reinit"

    # query params
    params = {
        "doc_id": collection_name,
        "memory_enabled": False
    }

    ### Before calling the reinit chat API tag user with doc in localhost 8000
    try:
        response = requests.post(url, params=params)

        # Raise exception for bad status codes
        response.raise_for_status()
        print(f"Chat reinitialized: {response.json()}")
        return response.json()

    except requests.exceptions.HTTPError as http_err:
        return {
            "status": "error",
            "message": f"HTTP error occurred: {http_err}",
            "response": response.text
        }

    except Exception as err:
        return {
            "status": "error",
            "message": f"Other error occurred: {err}"
        }



reinitialize_chat("777","auger_testing")
