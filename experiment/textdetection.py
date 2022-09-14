import sys
import os.path as osp
__dir__ = osp.dirname(osp.abspath(__file__))
PYTHON_PATH = osp.join(__dir__, '..')
sys.path.insert(0, PYTHON_PATH)
from modules.PaddleOCR.tools.infer import utility as paddle_ultility
from modules.PaddleOCR.tools.infer import predict_det2 as ppocr_det
from modules.PaddleOCR.ppocr.utils.utility import get_image_file_list
import cv2
import time
from shapely.geometry import Polygon, mapping
import numpy as np


def map_subtraction(img):
    height, width, _ = img.shape
    gray_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray_img.copy(), 200, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = filter(lambda cont: cv2.arcLength(cont, True) > 400, contours)
    contours = filter(lambda cont: cv2.contourArea(cont) > 100, contours)
    contours = filter(lambda cont: cv2.contourArea(cont) < (height-5)*(width-5), contours)

    rects = []
    for cont in contours:
        [x, y, w, h] = cv2.boundingRect(cont)
        rects.append([x, y, w, h])

    remove_rects = []
    for rect in rects:
        x = rect[0]
        y = rect[1]
        w = rect[2]
        h = rect[3]
        crop_img = gray_img[y:y+h, x:x+w]
        thresh_crop = cv2.threshold(crop_img, 250, 255, cv2.THRESH_BINARY)[1]
        white_num = 0
        black_num = 0
        for i in range(h):
            for j in range(w):
                if thresh_crop[i][j] == 255:
                    white_num += 1
                else:
                    black_num += 1
        if black_num < white_num:
            crop_gray_img2 = gray_img[y+int(0.03*h):y+int(0.97*h), x+int(0.03*w):x+int(0.97*w)]
            thresh_crop2 = cv2.threshold(crop_gray_img2.copy(), 200, 255, cv2.THRESH_BINARY_INV)[1]
            contours, _ = cv2.findContours(thresh_crop2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = filter(lambda cont: cv2.contourArea(cont, True) > (w*h)/50, contours)
            contours = filter(lambda cont: cv2.contourArea(cont) < 0.94*h*0.94*w, contours)
            if len(list(contours)) > 0:
                edges = cv2.Canny(crop_gray_img2.copy(), 100, 200, apertureSize=3)
                lines = cv2.HoughLinesP(edges.copy(), 1, np.pi / 180, 200, minLineLength=min(int(0.85 * w), int(0.85 * h)),
                                        maxLineGap=5)
                try:
                    if len(lines) >= 3:
                        remove_rects.append(rect)
                except TypeError:
                    continue
            else:
                remove_rects.append(rect)
        else:
            pass

    for remove_rect in remove_rects:
        rects.remove(remove_rect)

    if len(rects) != 0:
        for rect in rects:
            remove_area = img[rect[1] - 5:rect[1] + rect[3] + 5, rect[0] - 5:rect[0] + rect[2] + 5]
            mask = np.ones_like(remove_area.shape, np.uint8)
            img[rect[1] - 5:rect[1] + rect[3] + 5, rect[0] - 5:rect[0] + rect[2] + 5] = cv2.bitwise_not(mask, mask)
    else:
        pass

    return img


def get_boxes(img, args_paddle):
    text_detector = ppocr_det.TextDetector(args_paddle)
    dt_boxes, image_shape, _ = text_detector(img)
    dt_boxes = dt_boxes.astype(int)
    return img, dt_boxes, image_shape


def check_horizontal(box1, box2):
    horizontal = False
    center_height_box1 = int((box1[1][1] + box1[2][1])/2)
    center_height_box2 = int((box2[1][1] + box2[2][1])/2)
    height_box1 = abs(box1[1][1] - box1[2][1])
    height_box2 = abs(box2[1][1] - box2[2][1])
    if (center_height_box1 in range(box2[1][1], box2[2][1]) or center_height_box2 in range(box1[1][1], box1[2][1])) \
            and (max(height_box1, height_box2)/min(height_box1, height_box2)) < 1.5:
        if abs(center_height_box2 - center_height_box1) <= 0.25*max(height_box1, height_box2):
            horizontal = True
    return horizontal


def sorted_boxes(dt_boxes):
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    for i in range(num_boxes-1):
        for j in range(i+1, num_boxes):
            if check_horizontal(sorted_boxes[i], sorted_boxes[j])==True \
                    and sorted_boxes[j][0][0] < sorted_boxes[i][0][0]:
                tmp = sorted_boxes[i]
                sorted_boxes[i] = sorted_boxes[j]
                sorted_boxes[j] = tmp

    return np.array(sorted_boxes)


def expand_bounding_boxes(bounding_boxes, height, width):
    width_bb = np.linalg.norm(bounding_boxes[:, 0] - bounding_boxes[:, 1], axis=1)
    height_bb = np.linalg.norm(bounding_boxes[:, 0] - bounding_boxes[:, 3], axis=1)
    for i, width in enumerate(width_bb):
        width_bb[i]= int(width_bb[i])*1/100
    for i, height in enumerate(height_bb):
        height_bb[i] = int(height_bb[i])*20/100
    d_width = width_bb
    d_height = height_bb
    d_width = d_width.reshape((len(width_bb), 1))
    d_height = d_height.reshape((len(height_bb), 1))
    top_left = np.concatenate((-d_width, -d_height), axis=1)
    top_right = np.concatenate((d_width, -d_height), axis=1)
    bottom_right = np.concatenate((d_width, d_height), axis=1)
    bottom_left = np.concatenate((-d_width, d_height), axis=1)
    d_bb = np.concatenate([top_left, top_right, bottom_right, bottom_left], axis=1)
    d_bb = d_bb.reshape(len(bounding_boxes), 4, 2)
    bounding_boxes = bounding_boxes + d_bb
    return bounding_boxes.astype(int)


def crop2rec(img, boxes):
    croped_images = []
    for box in boxes:
        pts = np.array(box)
        # crop the bounding rect
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        croped_img = img[y:y+h, x:x+w]
        pts = pts - pts.min(axis=0)
        mask = np.zeros(croped_img.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        bg = np.ones_like(croped_img, np.uint8)*255
        cv2.bitwise_not(bg, bg, mask=mask)
        dst = cv2.bitwise_and(croped_img, croped_img, mask=mask)
        dst2 = bg + dst
        croped_images.append(dst2)
    return img, croped_images


def character_distance(thresh_img):
    height, width = thresh_img.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    img_erode = cv2.erode(thresh_img, kernel, iterations=10)
    center_y = int(height / 2)
    center_line = img_erode[center_y:center_y + 1, 0:width].flatten()
    gap_distances = []
    gap_ind = []
    start_point = 0
    for i in range(1, len(center_line) - 1):
        if center_line[i - 1] == 255 and center_line[i] == 0 and center_line[i + 1] == 0:
            start_point = i
        if center_line[i - 1] == 0 and center_line[i] == 0 and center_line[i + 1] == 255 and start_point != 0:
            end_point = i
            gap_ind.append(start_point)
            gap_distance = end_point - start_point
            gap_distances.append(gap_distance)

    if len(gap_distances) != 0:
        gap_distances = sorted(gap_distances)
        gap_median = np.median(np.array(gap_distances))
    else:
        gap_median = 50

    return gap_median


def get_gap_distance(img, polygon_cord):
    #crop img from polygon cordinates
    pts = np.array(polygon_cord)
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped_img = img[y:y + h, x:x + w]
    pts = pts - pts.min(axis=0)
    mask = np.zeros(croped_img.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    bg = np.ones_like(croped_img, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst = cv2.bitwise_and(croped_img, croped_img, mask=mask)
    dst2 = bg + dst
    # get_gap_distance
    img_2 = cv2.cvtColor(dst2.copy(), cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(img_2, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    height, width = img_2.shape
    #vertical_erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    img_erode = cv2.erode(thresh_img, kernel, iterations=10)
    #horizontal_erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    img_erode = cv2.erode(img_erode, kernel, iterations=2)
    #get gap distance
    center_y = int(height/2)
    center_line = img_erode[center_y:center_y+1, 0:width].flatten()
    gap_distances = []
    gap_ind = []
    start_point = 0
    for i in range(1, len(center_line)-1):
        if center_line[i-1]==0 and center_line[i]==255 and center_line[i+1]==255:
            start_point = i
        if center_line[i-1]==255 and center_line[i]==255 and center_line[i+1]==0 and start_point!=0:
            end_point = i
            gap_ind.append(start_point)
            gap_distance = end_point - start_point
            gap_distances.append(gap_distance)

    if len(gap_distances) != 0:
        gap_distances = sorted(gap_distances)
        gap_median = np.median(np.array(gap_distances))
        if gap_median < 5:
            new_gap_median = character_distance(thresh_img)
        else:
            new_gap_distances = [f for f in gap_distances if f >= gap_median]
            new_gap_median = np.median(np.array(new_gap_distances))
    else:
        new_gap_median = 50/7

    return new_gap_median


def merge_gap(dt_boxes, img):
    excess_ind = []
    excess_boxes = []
    for i in range(len(dt_boxes)-1):
        if i<(len(dt_boxes)-1):
            for j in range(i+1,len(dt_boxes)):
                if j<len(dt_boxes):
                    if check_horizontal(dt_boxes[i], dt_boxes[j]):
                        gap_distance1 = get_gap_distance(img, dt_boxes[i])
                        gap_distance2 = get_gap_distance(img, dt_boxes[j])
                        gap_distance = int(max(gap_distance1, gap_distance2))
                        gap_distance = min(7*gap_distance, 100)
                        if abs(dt_boxes[i][1][0]-dt_boxes[j][0][0]) <= gap_distance \
                                or abs(dt_boxes[i][2][0]-dt_boxes[j][3][0]) <= gap_distance:
                            new_box = np.array([[dt_boxes[i][0][0], min(dt_boxes[i][0][1], dt_boxes[j][0][1])], \
                                                [dt_boxes[j][1][0], min(dt_boxes[j][1][1], dt_boxes[i][1][1])], \
                                                [dt_boxes[j][2][0], max(dt_boxes[j][2][1], dt_boxes[i][2][1])], \
                                                [dt_boxes[i][3][0], max(dt_boxes[i][3][1], dt_boxes[j][3][1])]])
                            dt_boxes[i] = new_box
                            excess_ind.append(j)
                        else:
                            continue
                    else:
                        continue

                else:
                    break
        else:
            break
    excess_ind = set(excess_ind)
    for id in excess_ind:
        excess_boxes.append(dt_boxes[id].tolist())
    dt_boxes = dt_boxes.tolist()
    for box in excess_boxes:
        dt_boxes.remove(box)
    dt_boxes = np.array(dt_boxes)
    return dt_boxes



def overlap(dt_boxes):
    excess_ind = []
    excess_boxes = []
    for i in range(len(dt_boxes)-1):
        if i < (len(dt_boxes)-1):
            for j in range(i+1, len(dt_boxes)):
                if j < (len(dt_boxes)):
                    x10=dt_boxes[i][0][0]
                    y10=dt_boxes[i][0][1]
                    x11=dt_boxes[i][1][0]
                    y11=dt_boxes[i][1][1]
                    x12=dt_boxes[i][2][0]
                    y12=dt_boxes[i][2][1]
                    x13=dt_boxes[i][3][0]
                    y13=dt_boxes[i][3][1]
                    x20=dt_boxes[j][0][0]
                    y20=dt_boxes[j][0][1]
                    x21=dt_boxes[j][1][0]
                    y21=dt_boxes[j][1][1]
                    x22=dt_boxes[j][2][0]
                    y22=dt_boxes[j][2][1]
                    x23=dt_boxes[j][3][0]
                    y23=dt_boxes[j][3][1]
                    polygon1 = Polygon([(x10, y10), (x11,y11), (x12,y12), (x13, y13)])
                    polygon2 = Polygon([(x20, y20), (x21, y21), (x22, y22), (x23, y23)])
                    polygon3 = polygon1.intersection(polygon2)
                    if polygon3.is_empty == False:
                        if x10<x20:
                            if check_horizontal(dt_boxes[i], dt_boxes[j]):
                                new_box=np.array([[x10, min(y10,y20)], [x21, min(y21,y11)], [x22, max(y22,y12)],
                                                  [x13, max(y13,y23)]])
                                dt_boxes[i] = new_box
                                excess_ind.append(j)
                            else:
                                poly_mapped = mapping(polygon3)
                                if type(poly_mapped['coordinates'][0]) is tuple and len(poly_mapped['coordinates'][0])>2:
                                    poly_mapped = tuple(tuple(map(int, tup)) for tup in poly_mapped['coordinates'][0])
                                    sorted_poly = sorted(poly_mapped, key=lambda x: x[1])
                                    y_thresh = int((sorted_poly[0][1]+sorted_poly[-1][1])/2)
                                    dt_boxes[i][2][1] = y_thresh
                                    dt_boxes[i][3][1] = y_thresh
                                    dt_boxes[j][0][1] = y_thresh
                                    dt_boxes[j][1][1] = y_thresh
                                else:
                                    continue

                        else:
                            poly_mapped = mapping(polygon3)
                            if type(poly_mapped['coordinates'][0]) is tuple and len(poly_mapped['coordinates'][0]) > 2:
                                poly_mapped = tuple(tuple(map(int, tup)) for tup in poly_mapped['coordinates'][0])
                                sorted_poly = sorted(poly_mapped, key=lambda x: x[1])
                                y_thresh = int((sorted_poly[0][1] + sorted_poly[-1][1]) / 2)
                                dt_boxes[i][2][1] = y_thresh
                                dt_boxes[i][3][1] = y_thresh
                                dt_boxes[j][0][1] = y_thresh
                                dt_boxes[j][1][1] = y_thresh
                            else:
                                continue
                    else:
                        continue
                else:
                    break
        else:
            break
    excess_ind = set(excess_ind)
    for id in excess_ind:
        excess_boxes.append(dt_boxes[id].tolist())
    dt_boxes = dt_boxes.tolist()
    for box in excess_boxes:
        dt_boxes.remove(box)
    dt_boxes = np.array(dt_boxes)
    return dt_boxes


def text_detection(image_file_list, args_paddle):
    book_crop = []
    # id=0
    for image_file in image_file_list:
        img = cv2.imread(image_file)
        img = map_subtraction(img)
        img, dt_boxes, image_shape = get_boxes(img, args_paddle)
        dt_boxes = sorted_boxes(dt_boxes)
        height, width = image_shape[0:2]
        dt_boxes = expand_bounding_boxes(dt_boxes, height, width)
        dt_boxes = merge_gap(dt_boxes, img)
        dt_boxes = overlap(dt_boxes)   #######
        img, croped_images = crop2rec(img, dt_boxes)
        book_crop.append(croped_images)
        index = 0
        for crop in croped_images:
            cv2.imwrite(osp.join('../book/results2', str(index) + '.jpg'), crop)
            index += 1
        # for pts in dt_boxes:
        #     img = cv2.polylines(img, [pts], True, [0, 255, 0], 2)
        # cv2.imshow('img', cv2.resize(img, (800,1000)))
        # cv2.waitKey(0)


def main():
    st = time.time()
    args_paddle = paddle_ultility.parse_args()
    args_paddle.image_dir = "../book/gray_imgs1/image28.jpg"
    args_paddle.det_model_dir= "../models/PaddleOCR_det/det_db_inference"
    image_file_list = get_image_file_list(args_paddle.image_dir)
    text_detection(image_file_list=image_file_list, args_paddle=args_paddle)
    total_time = time.time() - st
    print(total_time)

if __name__ == "__main__":
    main()
