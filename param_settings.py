from time import sleep
import cv2
import numpy as np
import json
import sys
import os.path
from os import path
import random as rng
# import skimage.morphology as morphology

def nothing(x):
    pass

hsv_empty = {
    "H Lower" : 0,
    "H Higher" : 0,
    "S Lower" : 0,
    "S Higher" : 0,
    "V Lower" : 0,
    "V Higher" : 0,
}
circle_empty = {
    "min_radius" : 1,
    "max_radius" : 1,
    "param1" : 1,
    "param2" : 1,
    "param3" : 1,
    "param4" : 1,
}
transform_empty = {
    "object_canny_param1" : 50,
    "object_canny_param2" : 150,
    "object_morph_kernel" : 7,
    "field_canny_param1" : 50,
    "field_canny_param2" : 150,
    "field_morph_kernel" : 7,
}
object_hsv_path = 'vision/object_hsv.json'
field_hsv_path = 'vision/field_hsv.json'
circle_params_path = 'vision/circle_params.json'
transform_params_path = 'vision/transform_params.json'

if path.exists(object_hsv_path):
    with open(object_hsv_path, 'r') as openfile:
        object_hsv = json.load(openfile)
        # print("object_hsv exist")
else:
    object_hsv = hsv_empty
    # print("object_hsv does not exist")
    # print(object_hsv)

if path.exists(field_hsv_path):
    with open(field_hsv_path, 'r') as openfile:
        field_hsv = json.load(openfile)
else:
    field_hsv = hsv_empty

if path.exists(circle_params_path):
    with open(circle_params_path, 'r') as openfile:
        circle_params = json.load(openfile)
else:
    circle_params = circle_empty

if path.exists(transform_params_path):
    with open(transform_params_path, 'r') as openfile:
        transform_params = json.load(openfile)
else:
    transform_params = transform_empty

window_width = 300
window_height = 400

cv2.namedWindow('hsv_object')
cv2.resizeWindow('hsv_object', window_width, window_height)
cv2.createTrackbar('H Lower','hsv_object',object_hsv['H Lower'],179,nothing)
cv2.createTrackbar('H Higher','hsv_object',object_hsv['H Higher'],179,nothing)
cv2.createTrackbar('S Lower','hsv_object',object_hsv['S Lower'],255,nothing)
cv2.createTrackbar('S Higher','hsv_object',object_hsv['S Higher'],255,nothing)
cv2.createTrackbar('V Lower','hsv_object',object_hsv['V Lower'],255,nothing)
cv2.createTrackbar('V Higher','hsv_object',object_hsv['V Higher'],255,nothing)

cv2.namedWindow('hsv_field')
cv2.resizeWindow('hsv_field', window_width, window_height)
cv2.createTrackbar('H Lower','hsv_field',field_hsv['H Lower'],179,nothing)
cv2.createTrackbar('H Higher','hsv_field',field_hsv['H Higher'],179,nothing)
cv2.createTrackbar('S Lower','hsv_field',field_hsv['S Lower'],255,nothing)
cv2.createTrackbar('S Higher','hsv_field',field_hsv['S Higher'],255,nothing)
cv2.createTrackbar('V Lower','hsv_field',field_hsv['V Lower'],255,nothing)
cv2.createTrackbar('V Higher','hsv_field',field_hsv['V Higher'],255,nothing)

cv2.namedWindow('circle')
cv2.resizeWindow('circle', window_width, window_height)
cv2.createTrackbar('Min Radius','circle',circle_params['min_radius'],255,nothing)
cv2.createTrackbar('Max Radius','circle',circle_params['max_radius'],255,nothing)
cv2.createTrackbar('param1','circle',circle_params['param1'],255,nothing)
cv2.createTrackbar('param2','circle',circle_params['param2'],255,nothing)
cv2.createTrackbar('dp','circle',circle_params['param3'],255,nothing)
cv2.createTrackbar('minDist','circle',circle_params['param4'],255,nothing)

cv2.namedWindow('transform')
cv2.resizeWindow('transform', window_width, window_height)
cv2.createTrackbar('Object canny param1','transform',transform_params['object_canny_param1'],255,nothing)
cv2.createTrackbar('Object canny param2','transform',transform_params['object_canny_param2'],255,nothing)
cv2.createTrackbar('Object morph kernel','transform',transform_params['object_morph_kernel'],255,nothing)
cv2.createTrackbar('Field canny param1','transform',transform_params['field_canny_param1'],255,nothing)
cv2.createTrackbar('Field canny param2','transform',transform_params['field_canny_param2'],255,nothing)
cv2.createTrackbar('Field morph kernel','transform',transform_params['field_morph_kernel'],255,nothing)

# cv2.namedWindow('test')
# cv2.resizeWindow('test', 320, 240)
#read image
# img = cv2.imread("starbucks.jpg")
# if img is None:
#     sys.exit("Could not read the image.")

# cv2.imshow("Display window", img)
# k = cv2.waitKey(0)
# if k == ord("s"):
#     cv2.imwrite("starbucks.jpg", img)

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, 30)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
# camera.set(cv2.CAP_PROP_EXPOSURE, 200)

while(1):
    _,img = camera.read()
    if img is None:
        sleep(0.1)
        continue
    # img = cv2.flip(img,1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #object
    object_hsv['H Lower'] = cv2.getTrackbarPos('H Lower','hsv_object')
    object_hsv['H Higher'] = cv2.getTrackbarPos('H Higher','hsv_object')
    object_hsv['S Lower'] = cv2.getTrackbarPos('S Lower','hsv_object')
    object_hsv['S Higher'] = cv2.getTrackbarPos('S Higher','hsv_object')
    object_hsv['V Lower'] = cv2.getTrackbarPos('V Lower','hsv_object')
    object_hsv['V Higher'] = cv2.getTrackbarPos('V Higher','hsv_object')
    
    field_hsv['H Lower'] = cv2.getTrackbarPos('H Lower','hsv_field')
    field_hsv['H Higher'] = cv2.getTrackbarPos('H Higher','hsv_field')
    field_hsv['S Lower'] = cv2.getTrackbarPos('S Lower','hsv_field')
    field_hsv['S Higher'] = cv2.getTrackbarPos('S Higher','hsv_field')
    field_hsv['V Lower'] = cv2.getTrackbarPos('V Lower','hsv_field')
    field_hsv['V Higher'] = cv2.getTrackbarPos('V Higher','hsv_field')
    
    minRadius = cv2.getTrackbarPos('Min Radius','circle')
    maxRadius = cv2.getTrackbarPos('Max Radius','circle')
    param1 = cv2.getTrackbarPos('param1','circle')
    if param1 == 0:
        param1 = 1
    param2 = cv2.getTrackbarPos('param2','circle')
    if param2 == 0:
        param2 = 1
    param3 = cv2.getTrackbarPos('dp','circle')
    if param3 == 0:
        param3 = 1
    param4 = cv2.getTrackbarPos('minDist','circle')
    if param4 == 0:
        param4 = 1

    transform_params['object_canny_param1'] = cv2.getTrackbarPos('Object canny param1','transform')
    if transform_params['object_canny_param1'] == 0:
        transform_params['object_canny_param1'] = 1
    transform_params['object_canny_param2'] = cv2.getTrackbarPos('Object canny param2','transform')
    if transform_params['object_canny_param2'] == 0:
        transform_params['object_canny_param2'] = 1
    transform_params['object_morph_kernel'] = cv2.getTrackbarPos('Object morph kernel','transform')
    if transform_params['object_morph_kernel'] % 2:
        transform_params['object_morph_kernel'] += 1

    transform_params['field_canny_param1'] = cv2.getTrackbarPos('Field canny param1','transform')
    if transform_params['field_canny_param1'] == 0:
        transform_params['field_canny_param1'] = 1
    transform_params['field_canny_param2'] = cv2.getTrackbarPos('Field canny param2','transform')
    if transform_params['field_canny_param2'] == 0:
        transform_params['field_canny_param2'] = 1
    transform_params['field_morph_kernel'] = cv2.getTrackbarPos('Field morph kernel','transform')
    if transform_params['field_morph_kernel'] % 2:
        transform_params['field_morph_kernel'] += 1

    circle_params = {
        "min_radius" : minRadius,
        "max_radius" : maxRadius,
        "param1" : param1,
        "param2" : param2,
        "param3" : param3,
        "param4" : param4,
    }

    with open(object_hsv_path, "w") as outfile:
        json.dump(object_hsv, outfile)
        
    with open(field_hsv_path, "w") as outfile:
        json.dump(field_hsv, outfile)

    with open(circle_params_path, "w") as outfile:
        json.dump(circle_params, outfile)

    with open(transform_params_path, "w") as outfile:
        json.dump(transform_params, outfile)

    object_kernel = np.ones((transform_params['object_morph_kernel'],transform_params['object_morph_kernel']),"uint8")

    #red
    object_LowerRegion = np.array([object_hsv['H Lower'],object_hsv['S Lower'],object_hsv['V Lower']],np.uint8)
    object_upperRegion = np.array([object_hsv['H Higher'],object_hsv['S Higher'],object_hsv['V Higher']],np.uint8)

    object_mask = hsv
    #morphological operations
    object_mask = cv2.morphologyEx(object_mask,cv2.MORPH_CLOSE, object_kernel)
    object_mask = cv2.morphologyEx(object_mask,cv2.MORPH_OPEN, object_kernel)
    
    #sub: dilate-erode
    # object_mask = cv2.dilate(object_mask, object_kernel,iterations=1)
    # object_mask = cv2.erode(object_mask, object_kernel, iterations=2)

    object_mask = cv2.inRange(object_mask,object_LowerRegion,object_upperRegion)

    #morphological operations
    # object_mask = cv2.morphologyEx(hsv,cv2.MORPH_OPEN, object_kernel)
    # object_mask = cv2.morphologyEx(hsv,cv2.MORPH_CLOSE, object_kernel)
    
    object_edges = cv2.Canny(object_mask, transform_params['object_canny_param1'], transform_params['object_canny_param2'])

    circles = cv2.HoughCircles(object_edges, cv2.HOUGH_GRADIENT, param3, param4,
                              param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    output_object=cv2.bitwise_and(img, img, mask = object_mask)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # print(circles)
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(output_object, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(output_object, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    #mask
    field_LowerRegion = np.array([field_hsv['H Lower'],field_hsv['S Lower'],field_hsv['V Lower']],np.uint8)
    field_upperRegion = np.array([field_hsv['H Higher'],field_hsv['S Higher'],field_hsv['V Higher']],np.uint8)

    field_mask = cv2.inRange(hsv,field_LowerRegion,field_upperRegion)

    field_kernel = np.ones((transform_params['field_morph_kernel'],transform_params['field_morph_kernel']),"uint8")
    
    #morphological operations
    field_mask = cv2.morphologyEx(field_mask,cv2.MORPH_OPEN, field_kernel)
    field_mask = cv2.morphologyEx(field_mask,cv2.MORPH_CLOSE, field_kernel)
    
    #sub: dilate-erode
    # field_mask = cv2.dilate(field_mask, field_kernel,iterations=1)
    # field_mask = cv2.erode(field_mask, field_kernel, iterations=2)
    
    field_edges = cv2.Canny(field_mask, transform_params['field_canny_param1'], transform_params['field_canny_param2'])
    
    contours, hierarchy = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_field=cv2.bitwise_and(img, img, mask = field_mask)
    output_contours_original = cv2.drawContours(output_field, contours, -1, (0, 0, 255), 3)
    
    # cnt = contours[0]
    # approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt,True), True)
    # output_contours_approx = cv2.drawContours(output_field, approx, -1, (0, 255, 0), 3)
    
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
    # Draw contours + hull results
    drawing = np.zeros((field_edges.shape[0], field_edges.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        # cv2.drawContours(drawing, contours, i, color)
        cv2.drawContours(drawing, hull_list, i, color)
        
    # hull = cv2.convexHull(cnt)
    # output_contours_hull = cv2.drawContours(output_field, hull, -1, (0, 255, 0), 3)
    output_contours_hull = cv2.bitwise_or(output_field, drawing)
    
    enclosed_mask = np.zeros(img.shape, np.uint8)
    enclosed_mask = cv2.drawContours(enclosed_mask, hull_list, -1, (255,255,255), -1)
    # enclosed_mask = morphology.convex_hull_image(drawing)
    output_enclosed_mask = cv2.bitwise_and(img, enclosed_mask)
    
    output_final = cv2.bitwise_and(img, enclosed_mask, mask = object_mask)
    
    final_edges = cv2.Canny(output_final, 50, 150)

    circles = cv2.HoughCircles(final_edges, cv2.HOUGH_GRADIENT, param3, param4,
                              param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    # output_final=cv2.bitwise_and(img, img, mask = final_mask)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # print(circles)
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(output_final, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(output_final, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow("Object",output_object)
    cv2.imshow("Final",output_final)
    # cv2.imshow("Field",output_field)
    # cv2.imshow("Contours Original",output_contours_original)
    # cv2.imshow("Contours Approx",output_contours_approx)
    cv2.imshow("Contours Hull",output_contours_hull)
    # cv2.imshow("Enclosed Contours Hull",enclosed_mask)
    cv2.imshow("Output Enclosed Contours Hull",output_enclosed_mask)
    cv2.imshow("Canny", object_edges)
    cv2.imshow("Field Canny", field_edges)
    # cv2.imshow("Original", img)
    # cv2.imshow("Ball detect ",circles)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        camera.release()
        cv2.destroyAllWindows()
        break