import os
import urllib
import numpy as np
import pandas as pd
import multiprocessing
import logging
import time
import re
from PIL import Image
from tqdm import tqdm
from constants import allowed_units

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def common_mistake(unit):
    if unit in allowed_units:
        return unit
    if unit.replace('ter', 'tre') in allowed_units:
        return unit.replace('ter', 'tre')
    if unit.replace('feet', 'foot') in allowed_units:
        return unit.replace('feet', 'foot')
    if unit == 'lbs':
        return 'pound'  # Fix 'lbs' to 'pound'
    return unit

def parse_string(s):
    s_stripped = "" if s is None or str(s) == 'nan' else s.strip()
    if s_stripped == "":
        return None, None
    # Pattern to check if the format is valid: number followed by a unit
    pattern = re.compile(r'^-?\d+(\.\d+)?\s+[a-zA-Z\s]+$')
    if not pattern.match(s_stripped):
        logging.error(f"Invalid format found: {s}")
        raise ValueError(f"Invalid format in {s}")
    
    # Split the string into number and unit
    parts = s_stripped.split(maxsplit=1)
    number = float(parts[0])
    unit = common_mistake(parts[1])
    
    # Validate if the unit is in the allowed units
    if unit not in allowed_units:
        logging.error(f"Invalid unit [{unit}] in {s}. Allowed units: {allowed_units}")
        raise ValueError(f"Invalid unit [{unit}] found in {s}")
    
    return number, unit

def create_placeholder_image(image_save_path):
    try:
        placeholder_image = Image.new('RGB', (100, 100), color='black')
        placeholder_image.save(image_save_path)
    except Exception as e:
        logging.error(f"Error creating placeholder image: {e}")

def download_image(image_link, save_folder, retries=3, delay=3):
    if not isinstance(image_link, str):
        return

    filename = os.path.basename(image_link)
    image_save_path = os.path.join(save_folder, filename)

    if os.path.exists(image_save_path):
        return

    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(image_link, image_save_path)
            # Check if the downloaded image is valid
            with Image.open(image_save_path) as img:
                img.verify()  # Check for corrupted images
            return
        except Exception as e:
            logging.warning(f"Failed to download {image_link} (Attempt {attempt+1}/{retries}): {e}")
            time.sleep(delay)

    # If download fails, create a placeholder image
    logging.error(f"Failed to download {image_link} after {retries} attempts. Creating placeholder.")
    create_placeholder_image(image_save_path)

def download_images(image_links, download_folder, allow_multiprocessing=True):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    if allow_multiprocessing:
        with multiprocessing.Pool(min(64, multiprocessing.cpu_count())) as pool:
            # Use lambda to pass additional arguments to download_image
            list(tqdm(pool.imap(lambda link: download_image(link, download_folder), image_links), total=len(image_links)))
            pool.close()
            pool.join()
    else:
        for image_link in tqdm(image_links, total=len(image_links)):
            download_image(image_link, save_folder=download_folder, retries=3, delay=3)
