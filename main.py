from exif import Image
from datetime import datetime
import cv2
import math
import os
import time
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

    for i in range(10) : #may cause a problem that the code will be stuck here

        gyro, _ = imu_data()
        rotation = rotation_score(gyro)

        if rotation < ROTATION_THRESHOLD:
            print(f"{filename} accepted, rotation ok")
            return camera.take_photo(filename) # TOEGEVOEGD: maak daadwerkelijk foto
        else:
            print(f"{filename} rejected, rotation too high - retry")
        
def filter(distances,count): # Takes a list of distances, sorts it and gets the avarage distances of the middel 10 items.
    distance = 0
    length = len(distances)
    distances.sort()

    for i in range(int(length/2)-int(count/2),int(length/2)+int(count/2)):
        distance += distances[i]

    return distance/10
    
        

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

start_time = datetime.now()


from picamzero import Camera

camera = Camera()

start = datetime.now()



image_1 = capture_stable_image("photo1.jpg")
image_2 = capture_stable_image("photo2.jpg")  

speed_list = []
images = []

while (datetime.now()-start_time).total_seconds()  < 60: #600 sec
    
    image_1 = capture_stable_image("photo1.jpg")

    if (datetime.now()-start).total_seconds()  > 30: # it takes 45 for the space station to do change the angel with 1° so 2/3 of that is enough diffrence to calculate the distances.
        image_2 =  capture_stable_image("photo2.jpg") 
        start = datetime.now()

    else:
        
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

            if speed_formatted != 0:
                speed_list.append(speed_formatted)

                avrage_speed = filter(speed_list,1)

                #print(avrage_speed)


                Write_to_file(str(int(avrage_speed)))
                print("Write to file")



        except:
            print("Te weinig featers.")

        image_1 = image_2
