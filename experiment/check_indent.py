import cv2
import os
import os.path as osp

img_fold_path = './book/images'
imgfiles = [f for f in os.listdir(img_fold_path) if f.find('.jpg')!=-1]
if not osp.exists('./book/gray_imgs'):
    os.mkdir('./book/gray_imgs')
save_path = './book/gray_imgs'
for i in imgfiles:
    img_path = osp.join(img_fold_path, i)
    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(osp.join(save_path, i[:-4]+'gray.jpg'), gray_img)