from exif import Image
from datetime import datetime
import cv2
import math
import os
import time
from picamera import PiCamera
from picamzero import Camera

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


image_1_cv, image_2_cv = convert_to_cv(image_1, image_2)
kp1, kp2, des1, des2 = calculate_features(image_1_cv, image_2_cv, 500)
matches = calculate_matches(des1, des2)


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

coordinates_1, coordinates_2 = find_matching_coordinates(kp1, kp2, matches)
average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
    for i in range(int(length/2)-int(count/2),int(length/2)+int(count/2)):
        distance += distances[i]

    return distance/10

#average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)


def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    distance = feature_distance * GSD / 100000
    return distance / time_difference

def format_speed(speed, digits):
    exponent = int(math.floor(math.log10(abs(speed))))
    decimals = digits - 1 - exponent
    return f"{speed:.{int(decimals)}f}"  # Gecorrigeerd: f-string met correct formatting


start_time = datetime.now()


from picamzero import Camera

cam = Camera()



image_1 = cam.take_photo("photo1.jpg")
image_2 = cam.take_photo("photo2.jpg")  

while (datetime.now()-start_time).total_seconds()  < 60: #600 sec
    
    image_1 = cam.take_photo("photo1.jpg")

    time.sleep(45)

    image_2 = cam.take_photo("photo2.jpg")

    images = []

    speed_list = []
        
    time_difference = get_time_difference(image_1, image_2)

    image_1_cv, image_2_cv = convert_to_cv(image_1, image_2)

    keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000)
    matches = calculate_matches(descriptors_1, descriptors_2)

    try:

        coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)

        average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)

        GSD = 12648
        speed = calculate_speed_in_kmps(average_feature_distance, GSD, time_difference)
        speed_formatted = format_speed(speed, 5)

        speed_list.append(speed_formatted)

        avrage_speed = filter(speed_list,1)

        #print(avrage_speed)

        with open('result.txt', 'w') as file:
           file.write(avrage_speed+"km/s")



    except:
        print("Te weinig featers.")

    image_1 = image_2
