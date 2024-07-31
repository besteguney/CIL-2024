import requests
from PIL import Image
from io import BytesIO
import numpy as np
import os
from cities import city_coordinates

KEY = "INSERT YOUR API KEY HERE"
assert KEY

IMG_RES = 400
PIXEL_TRIM = 20
MIN_ROAD_COVERAGE = 10.0
MAP_ZOOM = 18
STEP_SIZE = 0.035
BASE_DIR = "INSERT YOUR DESTINATION FOLDER"

def retrieve_image_from_api(coords, map_kind, custom_styles=""):
    base_url = "https://maps.googleapis.com/maps/api/staticmap?"
    lon, lat = coords
    img_size = f"{IMG_RES + PIXEL_TRIM}x{IMG_RES + PIXEL_TRIM}"
    url = f"{base_url}center={lon},{lat}&zoom={MAP_ZOOM}&size={img_size}&format=PNG&maptype={map_kind}&key={KEY}{custom_styles}"
    
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"Fetch Error: Status {resp.status_code}")
        return None

    return Image.open(BytesIO(resp.content))

def store_cropped_image(img, directory, img_id, suffix):
    width, height = img.size
    cropped_img = img.crop((0, 0, width - PIXEL_TRIM, height - PIXEL_TRIM))

    if not os.path.exists(os.path.join(BASE_DIR, directory)):
        os.makedirs(os.path.join(BASE_DIR, directory))
    cropped_img.convert("RGB").save(os.path.join(BASE_DIR,directory,f"{img_id}{suffix}.png"))

def download_satellite_image(coords, directory, img_id):
    img = retrieve_image_from_api(coords, "satellite")
    if img:
        store_cropped_image(img, directory, img_id, "")

def generate_street_labels(coords, directory, img_id):
    styles = "&style=feature:all|element:labels|visibility:off" + \
             "&style=feature:administrative|visibility:off" + \
             "&style=feature:landscape|visibility:off" + \
             "&style=feature:poi|visibility:off" + \
             "&style=feature:water|visibility:off" + \
             "&style=feature:transit|visibility:off" + \
             "&style=feature:road|element:geometry|color:0xffffff"

    img = retrieve_image_from_api(coords, "roadmap", styles)
    if not img:
        return 0.0

    img_array = np.array(img)
    img_array[img_array != 0] = 255
    road_img = Image.fromarray(img_array)
    store_cropped_image(road_img, directory, img_id, "_label")

    width, height = road_img.size
    road_percentage = np.count_nonzero(img_array) * 100.0 / ((width - PIXEL_TRIM) * (height - PIXEL_TRIM))
    return road_percentage

def process_region(region, folder_id):
    start_lon, end_lon, start_lat, end_lat = region
    folder_name = f"LOCATION_{folder_id}"
    step_lon, step_lat = 0.015, 0.015
    img_id = 1

    total_scans = int((abs(start_lon - end_lon) / step_lon) * (abs(start_lat - end_lat) / step_lat))
    print(f"Initiating scan for {folder_name}, total files: {total_scans}")

    lon = start_lon
    while lon <= end_lon:
        lat = start_lat
        while lat <= end_lat:
            road_coverage = generate_street_labels((lon, lat), folder_name, img_id)
            if road_coverage > MIN_ROAD_COVERAGE:
                download_satellite_image((lon, lat), folder_name, img_id)
                img_id += 1
            lat += step_lat
            if img_id % 10 == 0:
                print(f"Processed {img_id} images for {folder_name}")
        lon += step_lon

def scan_locations(locations):
    for folder_id, region in enumerate(locations):
        process_region(region, folder_id)
    print("Data collection completed successfully.")

if __name__ == '__main__':
    scan_locations(city_coordinates)

