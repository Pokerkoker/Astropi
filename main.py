from exif import Image
from datetime import datetime
import cv2
import math
import os
import time
from picamzero import Camera
import numpy as np

# image_1 = 'imgs/photo_393_53245738275_o.jpg'
# image_2 = 'imgs/photo_394_53245245041_o.jpg'

def get_time(image):
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return time

def get_time_difference(image_1, image_2):
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    time_difference = time_2 - time_1

    return time_difference.seconds

#time_difference = get_time_difference(image_1, image_2)

def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    return image_1_cv, image_2_cv

def calculate_features(image_1_cv, image_2_cv, feature_number):
    orb = cv2.ORB_create(nfeatures = feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2

def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

#image_1_cv, image_2_cv = convert_to_cv(image_1, image_2)

#keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000)
#matches = calculate_matches(descriptors_1, descriptors_2)

def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
    match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:100], None)
    resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
    cv2.imshow('matches', resize)
    cv2.waitKey(0)
    cv2.destroyWindow('matches')

#display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches)

def find_matching_coordinates(keypoints_1, keypoints_2, matches):
        coordinates_1 = []
        coordinates_2 = []
        for match in matches:
            image_1_idx = match.queryIdx
            image_2_idx = match.trainIdx
            (x1,y1) = keypoints_1[image_1_idx].pt
            (x2,y2) = keypoints_2[image_2_idx].pt
            coordinates_1.append((x1,y1))
            coordinates_2.append((x2,y2))
        return coordinates_1, coordinates_2

#coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)

def calculate_mean_distance(coordinates_1, coordinates_2):
    all_distances = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    distances = []
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        distances.append(distance)
    return filter(distances,10)

def filter(distances,count): # Takes a list of distances, sorts it and gets the avarage distances of the middel 10 items.
    distance = 0
    length = len(distances)
    distances.sort()

    for i in range(int(length/2)-int(count/2),int(length/2)+int(count/2)):
        distance += distances[i]

    return distance/10

#average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)

def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    distance = feature_distance * GSD / 100000 # conversie van pixels naar km
    speed = distance / time_difference
    return speed

def format_speed(speed, nr_of_digits):
    exponent = int(math.floor(math.log10(abs(speed))))
    decimals = nr_of_digits - 1 - exponent
    speed_formatted = "{:.{}f}".format(speed, decimals)

    return speed_formatted

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

def calculate_land_matches(image_1_cv, image_2_cv):
    """
    Replacement for calculate_features + calculate_matches
    Returns: keypoints_1, keypoints_2, correct_matches
    """

    sift = cv2.SIFT_create(
        nfeatures=1000,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6
    )

    # Land masks
    mask1 = get_land_mask(cv2.cvtColor(image_1_cv, cv2.COLOR_GRAY2BGR))
    mask2 = get_land_mask(cv2.cvtColor(image_2_cv, cv2.COLOR_GRAY2BGR))

    kp1, des1 = sift.detectAndCompute(image_1_cv, mask1)
    kp2, des2 = sift.detectAndCompute(image_2_cv, mask2)

    if des1 is None or des2 is None:
        return [], [], []

    # FLANN matcher (required for SIFT)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches_knn = flann.knnMatch(des1, des2, k=2)

    # Strict Lowe ratio test
    good_matches = [m for m, n in matches_knn if m.distance < 0.5 * n.distance]

    if len(good_matches) < 6:
        return kp1, kp2, []

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    _, ransac_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)

    if ransac_mask is None:
        return kp1, kp2, []

    # Keep only RANSAC inliers
    correct_matches = [
        m for m, inlier in zip(good_matches, ransac_mask.ravel())
        if inlier
    ]

    return kp1, kp2, correct_matches

start_time = datetime.now()

cam = Camera()

image_1 = cam.take_photo("photo1.jpg")
image_2 = cam.take_photo("photo2.jpg")



total_time = 600 #the time that the program runs, normaly 600 sec

while (datetime.now() - start_time).total_seconds() < total_time:
    
    image_1 = cam.take_photo("photo1.jpg")

    # if total_time-(datetime.now() - start_time).total_seconds() > 50:#voorkomt dat de code voor 45 seconde wacht terwijl de 10 minuten bijna om zijn. and 
    #     time.sleep(45)#mogelijk nodig voor te wachten dat de foto's ver genoeg van elkaar zijn.
    # else:
    #     # Tijd bijna om → niet meer slapen
    #     pass 
        

    image_2 = cam.take_photo("photo2.jpg")

    images = []

    speed_list = []
        
    time_difference = get_time_difference(image_1, image_2)

    image_1_cv, image_2_cv = convert_to_cv(image_1, image_2)

    keypoints_1, keypoints_2, matches = calculate_land_matches(image_1_cv, image_2_cv)

    try:

        coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)

        average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)

        GSD = 12648
        speed = calculate_speed_in_kmps(average_feature_distance, GSD, time_difference)
        speed_formatted = format_speed(speed, 5)

        speed_list.append(speed_formatted)

        avrage_speed = filter(speed_list,1)

        #print(avrage_speed)

        Write_to_file(f"{avrage_speed} km/s")
        #Write_to_file(f"{(datetime.now()-start_time).total_seconds()}")



    except:
        print("Te weinig featers.")

    image_1 = image_2

    Write_to_file(f"{(datetime.now()-start_time).total_seconds()}")

    break

    #image_2 = cam.take_photo("photo2.jpg")
