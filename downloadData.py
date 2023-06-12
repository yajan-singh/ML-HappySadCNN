import os
from bing_image_downloader import downloader

Queries = ["Happy People", "Sad People", "Angry People",
           "Surprised People", "Disgusted People", "Fearful People"]

# Downloading Images
for query in Queries:
    downloader.download(query, limit=100,  output_dir='dataset')

# Renaming Folders
for dir in os.listdir('dataset'):
    os.rename('dataset/' + dir, 'dataset/' + dir.split(' ')[0].upper())
