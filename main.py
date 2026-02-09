from exif import Image
from datetime import datetime
import cv2
import math
from sense_hat import SenseHat
from picamera import PiCamera
from time import sleep

sensor = SenseHat()
camera = PiCamera()

image_1 = 'image1.jpg'
image_2 = 'image2.jpg'

ROTATION_FOUT = 0.1


def imu_data():
    gyro = sensor.get_gyroscope_raw()
    accel = sensor.get_accelerometer_raw()
    return gyro, accel

def rotation_score(gyro):
    return abs(gyro['x']) + abs(gyro['y']) + abs(gyro['z'])


def capture_stable_image(filename):
   
    gyro, _ = imu_data()
    rotation = rotation_score(gyro)

   
    if rotation < ROTATION_FOUT:
        print(f"{filename} accepted, rotation ok")
    else:
        print(f"{filename} rejected, rotation too high - retry")
        

capture_stable_image(image_1)
sleep(5)
capture_stable_image(image_2)


def get_time(image):
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return time

def get_time_difference(image_1, image_2):
    return (get_time(image_2) - get_time(image_1)).seconds

time_difference = get_time_difference(image_1, image_2)


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
    return distances[int(len(distances)/2)]

def calculate_mean_distance(c1, c2):
    distances = []
    for p1, p2 in zip(c1, c2):
        distances.append(math.hypot(p1[0]-p2[0], p1[1]-p2[1]))
    return median(distances)

coordinates_1, coordinates_2 = find_matching_coordinates(kp1, kp2, matches)
average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)


def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    distance = feature_distance * GSD / 100000
    return distance / time_difference

def format_speed(speed, digits):
    exponent = int(math.floor(math.log10(abs(speed))))
    decimals = digits - 1 - exponent
    return "{:.{}f}".format(speed, decimals)


GSD = 12648
speed = calculate_speed_in_kmps(average_feature_distance, GSD, time_difference)
speed_formatted = format_speed(speed, 5)

print(speed_formatted)

with open('result.txt', 'w') as file:
    file.write(speed_formatted)
