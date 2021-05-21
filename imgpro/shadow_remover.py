import cv2
import numpy as np

SAVE_PREFIX ="/home/work/trainsets/v05_single_lighting_or_shadow/"

img_name = SAVE_PREFIX+"20201130_000309568-20_1.jpg"
saved_name = SAVE_PREFIX + "20201130_000309568-20_1_shadow.jpg"
saved_name_norm = SAVE_PREFIX + "20201130_000309568-20_1_shadow_norm.jpg"

img = cv2.imread(img_name, -1)

rgb_planes = cv2.split(img)

result_planes = []
result_norm_planes = []
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)

result = cv2.merge(result_planes)
result_norm = cv2.merge(result_norm_planes)

cv2.imwrite(saved_name, result)
cv2.imwrite(saved_name_norm, result_norm)
