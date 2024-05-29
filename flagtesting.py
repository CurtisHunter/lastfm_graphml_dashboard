import requests
from PIL import Image
from io import BytesIO

# Define the flag URLs for the countries you want to display
flag_urls = {
    "India": "https://flagsapi.com/IN/flat/64.png",
    "China": "https://flagsapi.com/CN/flat/64.png",
    "Indonesia": "https://flagsapi.com/ID/flat/64.png",
    # Add more countries as needed
}

# Create a dictionary to store the flags
flags = {}

# Fetch and store the flags
for country, url in flag_urls.items():
    response = requests.get(url)
    flags[country] = Image.open(BytesIO(response.content)).resize((40, 30), Image.Resampling.LANCZOS)

# Now you can use the flags dictionary to access the flag images for each country
# For example, to display the flag for India:
flags["India"].show()