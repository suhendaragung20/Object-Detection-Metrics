
from utils import visualization_utils as vis_util
import numpy as np
import cv2
from detector import detector
from imutils import paths
import os
import time
from utils import plot_cv

import time

import argparse




list_voc = []

def write_to_voc(list_voc):
    
    idx = 1

    for voc in list_voc:
        voc_string_txt = ''
        
        (bboxes, names, scores, (h, w), target_save_image_path) = voc

        filename_without_extension = os.path.splitext(target_save_image_path)[0]

        for box, name, score in zip(bboxes, names, scores):
            (xmin, ymin, xmax, ymax) = box

            if xmax >= w:
                xmax = w - 1
            if ymax >= h:
                ymax = h - 1

            voc_string_txt = voc_string_txt + name + ' .' + str(score) + ' ' \
                                                    + str(int(xmin)) + ' ' \
                                                    + str(int(ymin)) + ' ' \
                                                    + str(int(xmax)) + ' ' \
                                                    + str(int(ymax)) + '\n'

        with open('detections/' + str(filename_without_extension) + '.txt', 'w') as f: 
            f.write("%s\n" % voc_string_txt)

        idx += 1

        # with open('predict/' + filename_without_extension + '.txt', 'w') as f: 
        #     f.write("%s\n" % voc_string_txt)


det = detector.detector()

name_class = ["plate", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", 
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", 
                "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", 
                "W", "X", "Y", "Z"]

image_folder = 'data_voc'

#imagePaths = paths.list_images(image_folder)
imageNames = os.listdir(image_folder)


count = 1

# loop over the image paths
for imageName in imageNames:

    #print(imageName)


    image = cv2.imread(image_folder + '/' + imageName)

    h, w = image.shape[:2]

    (boxes, scores, classes, num, category_index) = det.detect_plate(image)

    if len(boxes) == 0:
        print('skip detection')
        continue

    boxes = boxes[0]
    scores = scores[0]
    classes = classes[0]

    predict_boxes = []
    predict_names = []
    predict_scores = []

    for box, score, class_idx in zip(boxes, scores, classes):
        (startY, startX, endY, endX) = box

        startX = int(startX * w)
        startY = int(startY * h)
        endX = int(endX * w)
        endY = int(endY * h)

        box = (startX, startY, endX, endY)

        text = name_class[int(class_idx-1)]

        if score > 0.7:
            predict_boxes.append(box)
            predict_names.append(text)
            predict_scores.append(int(score*100))


    list_voc.append((predict_boxes, predict_names, predict_scores, (h,w), imageName))

    print(count, "======================")    

    count += 1

write_to_voc(list_voc)

# Clean up
cv2.destroyAllWindows()