import os
import json
import requests
from tqdm import tqdm

# Directory paths
json_dir = './midjourney'
output_dir = 'images'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Function to download images
def download_image(url, filename):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded {url} to {filename}")
        else:
            pass
            #print(f"Failed to download {url}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")


# Iterate over all JSON files in the specified directory
for json_file in os.listdir(json_dir):
    if json_file.endswith('.json'):
        json_file_path = os.path.join(json_dir, json_file)
        print(f"Processing {json_file_path}...")

        # Load each JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Loop through messages in each JSON file and download images
        for message_group in tqdm(data['messages']):
            for message in message_group:
                attachments = message.get('attachments', [])
                for attachment in attachments:
                    img_url = attachment['url']
                    img_filename = os.path.join(output_dir, attachment['filename'])
                    download_image(img_url, img_filename)

print("All images downloaded successfully!")
