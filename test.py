from exif import Image
from datetime import datetime
import cv2
import math
import os
import time
import numpy as np
from picamera import PiCamera
from picamzero import Camera
from sense_hat import SenseHat

# image_1 = 'imgs/photo_393_53245738275_o.jpg'
# image_2 = 'imgs/photo_394_53245245041_o.jpg'

def get_time(image):
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return time

def get_time_difference(image_1, image_2):
    return (get_time(image_2) - get_time(image_1)).seconds

#time_difference = get_time_difference(image_1, image_2)


def convert_to_cv(image_1, image_2):
    return cv2.imread(image_1, 0), cv2.imread(image_2, 0)

def calculate_features(image_1_cv, image_2_cv, feature_number):
    orb = cv2.ORB_create(nfeatures=feature_number)
    kp1, des1 = orb.detectAndCompute(image_1_cv, None)
    kp2, des2 = orb.detectAndCompute(image_2_cv, None)
    return kp1, kp2, des1, des2

def calculate_matches(descriptors_1, descriptors_2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)
    return sorted(matches, key=lambda x: x.distance)


def find_matching_coordinates(kp1, kp2, matches):
    c1, c2 = [], []
    for m in matches:
        (x1, y1) = kp1[m.queryIdx].pt
        (x2, y2) = kp2[m.trainIdx].pt
        c1.append((x1, y1))
        c2.append((x2, y2))
    return c1, c2

def median(distances):
    distances.sort()
    return distances[len(distances) // 2]  # Gecorrigeerd: integer division

def calculate_mean_distance(c1, c2):
    distances = []
    for p1, p2 in zip(c1, c2):
        distances.append(math.hypot(p1[0]-p2[0], p1[1]-p2[1]))
    return median(distances)

def imu_data():
    sensor = SenseHat()
    gyro = sensor.get_gyroscope_raw()
    accel = sensor.get_accelerometer_raw()
    return gyro, accel

def rotation_score(gyro):
    return abs(gyro['x']) + abs(gyro['y']) + abs(gyro['z'])


def capture_stable_image(filename):
    """Capture image when rotation is stable"""
    ROTATION_THRESHOLD = 0.1

    gyro, _ = imu_data()
    rotation = rotation_score(gyro)

    if rotation < ROTATION_THRESHOLD:
        print(f"{filename} accepted, rotation ok")
        return camera.take_photo(filename) # TOEGEVOEGD: maak daadwerkelijk foto
    else:
        print(f"{filename} rejected, rotation too high - retry")
        return None
        
def filter(distances): # Takes a list of distances, sorts it and gets the avarage distances of the middel 10 items.
    distance = 0
    mid = len(distances)//2
    distances.sort()

    if not distances:
        return None

    if len(distances) % 2 == 1:
        return distances[mid]
    else:
        return (distances[mid - 1] + distances[mid]) / 2
    
        

def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    distance = feature_distance * GSD / 100000
    return distance / time_difference

def format_speed(speed, digits):
    print("Tdif"+str(speed))
    exponent = int(math.floor(math.log10(abs(speed))))
    decimals = digits - 1 - exponent
    return f"{speed:.{int(decimals)}f}"  # Gecorrigeerd: f-string met correct formatting

def Write_to_file(data):
    with open('result.txt', 'w') as file:
           file.write(data)

def get_land_mask(img):
    # HSV kleurfiltering blijft de beste basis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1] 
    v = hsv[:, :, 2]

    # Strengere kleurselectie: land heeft meestal meer verzadiging dan oceaan/wolken
    _, color_mask = cv2.threshold(s, 35, 255, cv2.THRESH_BINARY)
    # Donkere gebieden filteren (diepe oceaan uitsluiten)
    _, brightness_mask = cv2.threshold(v, 30, 255, cv2.THRESH_BINARY)
    
    combined_mask = cv2.bitwise_and(color_mask, brightness_mask)
    
    # Haal kleine ruis weg
    kernel = np.ones((7,7), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    return combined_mask


def process_land(img1, img2):
    """
    Detecteert SIFT features op land, matcht ze met FLANN,
    filtert op ratio test en RANSAC inliers.
    
    Return:
        inlier_matches : lijst van cv2.DMatch objects (RANSAC inliers)
        kp1, kp2       : keypoints van beide afbeeldingen
    """
    if img1 is None or img2 is None:
        return None

    # SIFT
    sift = cv2.SIFT_create(
        nfeatures=1000,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6
    )

    # FLANN matcher
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=50)
    )

    # Land mask
    mask1 = get_land_mask(img1)
    mask2 = get_land_mask(img2)

    # Keypoints & descriptors
    kp1, des1 = sift.detectAndCompute(img1, mask1)
    kp2, des2 = sift.detectAndCompute(img2, mask2)

    if des1 is None or des2 is None or len(kp1) < 10:
        return None

    # KNN match
    matches = flann.knnMatch(des1, des2, k=2)

    # Zeer strenge ratio test
    good_matches = [m for m, n in matches if m.distance < 0.5 * n.distance]

    if len(good_matches) < 6:
        return None

    # RANSAC filtering
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)

    if mask is None:
        return None

    # Filter op inliers
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]

    if len(inlier_matches) < 8:
        return None

    return inlier_matches, kp1, kp2



start_time = time.monotonic()


from picamzero import Camera

camera = Camera()

start = time.monotonic()



image_1 = capture_stable_image("photo1.jpg")
image_2 = capture_stable_image("photo2.jpg")  

speed_list = []
images = []

while time.monotonic()-start_time  < 60: #600 sec
    
    image_1 = capture_stable_image("photo1.jpg")

    if image_1 is None:
        time.sleep(0.2)
        continue

    if (time.monotonic() - start) <= 30:
        time.sleep(0.2)
        continue # it takes 45 for the space station to do change the angel with 1° so 2/3 of that is enough diffrence to calculate the distances.

    image_2 = capture_stable_image("photo2.jpg")

    if image_2 is None:
        continue

    result = process_land(cv2.imread(image_1), cv2.imread(image_2))

    if result is None:
        time.sleep(0.2)
        continue

    inlier_matches, kp1, kp2 = result

    image_1_cv, image_2_cv = convert_to_cv(image_1, image_2)

    keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000)
    matches = calculate_matches(descriptors_1, descriptors_2)

    coords1 = [kp1[m.queryIdx].pt for m in inlier_matches]
    coords2 = [kp2[m.trainIdx].pt for m in inlier_matches]



    average_feature_distance = calculate_mean_distance(coords1, coords2)
    time_difference = get_time_difference(image_1, image_2)



    GSD = 12648
    speed = calculate_speed_in_kmps(average_feature_distance, GSD, time_difference)



    if speed != 0:
        speed_list.append(speed)

        avrage_speed = filter(speed_list)

        #print(avrage_speed)


        Write_to_file(format_speed(speed,2))
        print("Write to file")


    image_1 = image_2

    start = time.monotonic()

    time.sleep(0.2)
