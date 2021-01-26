import cv2
import datetime
import numpy as np



def press_key():
    key = cv2.waitKey(1) & 0xFF
    return key


def plot_left_area(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    startX = 0
    startY = 0
    endX = int(w*(1/2))
    endY = h
    color_bgr = (0, 0, 255)
    cv2.rectangle(overlay, (startX, startY), (endX, endY), color_bgr, -1)

    alpha = 0.05  # Transparency factor.

    # Following line overlays transparent rectangle over the image
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return frame


def plot_dilarang_parkir_area(frame, box, min_park, max_park):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    (startX, startY, endX, endY) = box
    box_w = endX - startX
    box_h = endY - startY
    box_center_x = int(box_w/2) + startX

    ratio_low = 1
    ratio_high = 3

    if box_center_x < int(w/2): 
        start_area_x = box_center_x - int(ratio_high*box_w)
        end_area_x = box_center_x + int(ratio_low*box_w)
    else:
        start_area_x = box_center_x - int(ratio_low*box_w)
        end_area_x = box_center_x + int(ratio_high*box_w)

    if start_area_x < 0:
        start_area_x = 0

    if end_area_x > w:
        end_area_x = w


    if start_area_x < min_park:
        min_park = start_area_x

    if end_area_x > max_park:
        max_park = end_area_x

    color_bgr = (0, 255, 0)
    cv2.rectangle(overlay, (start_area_x, startY), (end_area_x, h),
                  color_bgr, -1)

    alpha = 0.1  # Transparency factor.

    # Following line overlays transparent rectangle over the image
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


    return min_park, max_park, frame






def plot_object(frame, box, text, idx_color):

    if idx_color == 0:
        color_bgr = (0, 147, 0)
    elif idx_color == 1:
        color_bgr = (147, 147, 0)
    elif idx_color == 2:
        color_bgr = (147, 0, 0)
    elif idx_color == 3:   
        color_bgr = (0, 147, 147)
    elif idx_color == 4:   
        color_bgr = (100, 0, 100)
    elif idx_color == 5:   
        color_bgr = (0, 0, 100)


    overlay = frame.copy()

    (H, W) = frame.shape[:2]

    if H > W:
        min_frame = W
    else:
        min_frame = H



    box_border = int(W / 400)

    #font_size = 0.3 

    font_size = min_frame / 3000

    (startX, startY, endX, endY) = box

    y = startY - 10 if startY - 10 > 10 else startY + 10

    yBox = y + 5

    cv2.rectangle(overlay, (startX, startY), (endX, endY),
                  (255, 255, 255), box_border+4)

    cv2.rectangle(overlay, (startX, startY), (endX, endY),
                  color_bgr, box_border+2)


    font_scale = 0.3
    font = cv2.FONT_HERSHEY_SIMPLEX

    # make a black image
    img = np.zeros((500, 500))
    # set some text
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=(0.4*box_border), thickness=box_border)[0]
    # set the text start position
    text_offset_x = startX
    text_offset_y = y
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
    cv2.rectangle(overlay, box_coords[0], box_coords[1], color_bgr, cv2.FILLED)
    cv2.putText(overlay, text, (text_offset_x, text_offset_y), font, fontScale=(0.4*box_border), color=(255, 255, 255), thickness=box_border)


    alpha = 0.6  # Transparency factor.

    # Following line overlays transparent rectangle over the image
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return frame


def plot_fps(frame, frame_rate_calc):
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    return frame


def clear_cv(vs):
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()