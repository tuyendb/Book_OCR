import cv2
import numpy as np
import os
import os.path as osp

def map_subtraction(img):
    height, width, _ = img.shape
    gray_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray_img.copy(), 200, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imshow('adas', cv2.resize(thresh_img, (800, 1000)))
    cv2.waitKey()
    contours, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = filter(lambda cont: cv2.arcLength(cont, True) > 400, contours)
    contours = filter(lambda cont: cv2.contourArea(cont) > 10000, contours)
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


if __name__ == '__main__':
    # img_dir = [f for f in os.listdir('../book/gray_imgs1') if f.find('.jpg')!=-1]
    # img_dir = sorted(img_dir, key = lambda x : x[5:])
    # img_dir = sorted(img_dir, key = lambda x : len(x[5:]))
    # print(len(img_dir))
    # i = 0
    # while True:
        # img1 = cv2.imread(osp.join('../book/gray_imgs1',img_dir[i]))
        # img = map_subtraction(img1)
        # cv2.namedWindow(img_dir[i])
        # cv2.moveWindow(img_dir[i], 1000, 0)
        # cv2.imshow(img_dir[i], cv2.resize(img, (800,1000)))
        #
        # if cv2.waitKey() & 0xFF == ord('n'):
        #     cv2.destroyAllWindows()
        #     i+=1
        #     if i == len(img_dir):
        #         break
        # if cv2.waitKey() & 0xFF == ord('q'):
        #     break
        # if cv2.waitKey() & 0xFF == ord('p'):
        #     cv2.destroyAllWindows()
        #     i-=1
        #     if i == 0:
        #         break

    img_file_path1 = '../book/van2_7/image144.jpg'
    img = cv2.imread(img_file_path1)
    cv2.imshow('asd', cv2.resize(map_subtraction(img), (800,1000)))
    cv2.waitKey()