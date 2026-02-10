import cv2
import numpy as np

img1 = cv2.imread(os.path.join(folder, ""))
img2 = cv2.imread(os.path.join(folder, ""))

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

def process_astro_land(img1, img2):
    
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

    mask1 = get_land_mask(img1)
    mask2 = get_land_mask(img2)

    # Detecteer kenmerken ENKEL op het gemaskeerde land
    kp1, des1 = sift.detectAndCompute(img1, mask1)
    kp2, des2 = sift.detectAndCompute(img2, mask2)

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
    return good_matches
process_astro_land(img1, img2)