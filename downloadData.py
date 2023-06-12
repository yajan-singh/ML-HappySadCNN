import imghdr
import os
from bing_image_downloader import downloader
import cv2


def download_images(queries, limit=100):
    # Downloading Images
    for query in queries:
        downloader.download(query, limit=limit,  output_dir='dataset')

    # Renaming Folders
    for dir in os.listdir('dataset'):
        os.rename('dataset/' + dir, 'dataset/' + dir.split(' ')[0].upper())


def remove_faulty_images():
    # Remove Faulty Images
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    for dir in os.listdir('dataset'):
        for file in os.listdir('dataset/' + dir):
            try:
                img = cv2.imread('dataset/' + dir + '/' + file)
                ext = imghdr.what('dataset/' + dir + '/' + file)
                if img is None or ext not in exts:
                    os.remove('dataset/' + dir + '/' + file)
            except:
                os.remove('dataset/' + dir + '/' + file)
