import cv2
import numpy as np
import os

folder_path = './imgs'
output_folder = 'output_matches55'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

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

def process_astro_land(folder):
    images = sorted([f for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    # SIFT UPGRADE:
    # contrastThreshold: negeer kenmerken in gebieden met weinig contrast (zoals zee/mist)
    # edgeThreshold: negeer kenmerken die op lijnen lijken (zoals wolkenranden)
    sift = cv2.SIFT_create(
        nfeatures=1000,
        contrastThreshold=0.04, 
        edgeThreshold=10,
        sigma=1.6
    )
    
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    for i in range(len(images) - 1):
        img1 = cv2.imread(os.path.join(folder, images[i]))
        img2 = cv2.imread(os.path.join(folder, images[i+1]))
        if img1 is None or img2 is None: continue

        mask1 = get_land_mask(img1)
        mask2 = get_land_mask(img2)

        # Detecteer kenmerken ENKEL op het gemaskeerde land
        kp1, des1 = sift.detectAndCompute(img1, mask1)
        kp2, des2 = sift.detectAndCompute(img2, mask2)

        if des1 is None or des2 is None or len(kp1) < 10:
            continue

        matches = flann.knnMatch(des1, des2, k=2)

        # Zeer strenge ratio test (0.5 ipv 0.7)
        # Dit zorgt dat alleen UNIEKE punten op land overblijven
        good_matches = [m for m, n in matches if m.distance < 0.5 * n.distance]

        inliers = 0
        status_mask = None

        if len(good_matches) > 6:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # RANSAC: Gebruik een kleinere threshold (2.0) om alleen exact lopende punten te houden
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)
            
            if mask is not None:
                inliers = np.sum(mask)
                status_mask = mask.ravel().tolist()

        # Alleen opslaan als we echt kwalitatieve inliers hebben
        if inliers > 8:
            print(f"Goede match op land: {images[i+1]} ({inliers} betrouwbare punten)")
            res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                                  matchesMask=status_mask, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            cv2.imwrite(os.path.join(output_folder, f"match_{images[i+1]}"), res)

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


process_astro_land(folder_path)

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
    