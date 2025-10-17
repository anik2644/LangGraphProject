import requests
import sys
import os
import json

product_name = sys.argv[1]  # The product name is passed as an argument
api_url = os.getenv('PRODUCT_API_URL', 'http://localhost:8080/api/products/name')

try:
    # Make the API request
    response = requests.get(f"{api_url}/{product_name}")
    if response.status_code == 200:
        # Return product details as JSON
        print(json.dumps({"product_details": response.json()}))
    else:
        print(json.dumps({"error": f"Product not found. Status code: {response.status_code}"}))
except Exception as e:
    print(json.dumps({"error": str(e)}))
