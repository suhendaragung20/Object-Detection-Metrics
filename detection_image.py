
from utils import visualization_utils as vis_util
import numpy as np
import cv2
from detector import detector
from imutils import paths
import os
import time



det = detector.detector()

image_folder = 'data_voc'

#imagePaths = paths.list_images(image_folder)
imageNames = os.listdir(image_folder)

# loop over the image paths
for imageName in imageNames:

    #print(imageName)

    image = cv2.imread(image_folder + '/' + imageName)
    print(image_folder + '/' + imageName)

    h, w = image.shape[:2]

    (boxes, scores, classes, num, category_index) = det.detect_plate(image)

    # print(category_index)

    if len(boxes) == 0:
        print('skip detection')
        continue

    b = boxes[0]
    for bb in b:
        # ymin, xmin, ymax, xmax
        (ymin, xmin, ymax, xmax) = bb
        print(int(xmin*w), int(ymin*h), int(xmax*w), int(ymax*h))
        # print((xmin, ymin, xmax, ymax))
    c = classes[0]
    print(c)


    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.70)

    # All the results have been drawn on image. Now display the image.
    cv2.namedWindow("detector", cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty("detector", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    cv2.imshow('detector', image)

    # Press any key to close the image
    cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
