from utils import character_distance, ImgRecognizer
from init_models import models_init
import numpy as np
import cv2


text_recognizer = models_init()[1]


class TextLine:

    def __init__(self, img, coord):
        self._img = img
        self._coord = coord
        self._binary_img = None
        self._text = None
        self._x_coords = None
        self._y_coords = None
        self._character_height = None
        self._character_thickness = None
        self._gray_lv = None
        self._text_height = None

    @property
    def img(self):
        return self._img

    @property
    def binary_img(self):
        if self._binary_img is None:
            gray_img = cv2.cvtColor(self.img.copy(), cv2.COLOR_RGB2GRAY)
            self._binary_img = cv2.threshold(gray_img.copy(), 200, 255, cv2.THRESH_BINARY)[1]
        return self._binary_img

    @property
    def top_left(self) -> tuple:
        return self._coord[0], self._coord[1]

    @property
    def bottom_right(self) -> tuple:
        return self._coord[0] + self._coord[2], self._coord[1] + self._coord[3]

    @property
    def box_height(self):
        return self._coord[3]

    @property
    def box_width(self):
        return self._coord[2]

    @property
    def text(self) -> str:
        if self._text is None:
            self._text = ImgRecognizer(self._img.copy(), text_recognizer).recognize
        return self._text
    
    @property
    def check_all_upper(self) -> bool:
        return self.text.isupper()

    @property
    def check_first_upper(self) -> bool:
        return self.text[0].isupper()

    @property
    def check_3all_uppper(self) -> bool:
        return self.text[:3].isupper()

    @property
    def check_not_subtitle_list(self):
        check_list = ['/', ')', '.', ':', '-', '?']
        is_subtitle = False
        for check in check_list:
            if check in self.text[:2]:
                is_subtitle = True
                break
        return not is_subtitle

    @property
    def check_unnecessary(self):
        unnecessary = ["BÀI", "Bài", "Bai", "BAI", "bài", "bai"]
        unncsr = False
        if 6 >= self.text.__len__() >= 3:
            for unnec1 in unnecessary:
                if unnec1 in self.text:
                    unncsr = True
                    break
        return unncsr

    @property
    def gray_level(self):
        if self._gray_lv is None:
            self._img = self._img.flatten()
            self._img = np.delete(self._img, np.where(self._img >= 200))
            self._gray_lv = np.median(self._img)
        return self._gray_lv

    @property
    def character_thickness(self):
        if self._character_thickness is None:
            self._character_thickness = round(character_distance(self.binary_img.copy()), 1)
        return self._character_thickness

    @property
    def y_coords(self) -> tuple:
        if self._y_coords is None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
            erode_img = cv2.erode(self.binary_img.copy(), kernel=kernel, iterations=2)
            y_below_coords = []
            y_above_coords = []
            for i in range(int(self.box_width)):
                vertical_line = erode_img[:, i]
                for j in range(1, self.box_height):
                    if vertical_line[-j] == 0:
                        y_below_coord = self.box_height - j
                        y_below_coords.append(y_below_coord)
                        break
                for j1 in range(0, self.box_height - 1):
                    if vertical_line[j1] == 0 and vertical_line[j1+1] == 0:
                        y_above_coord = j1
                        y_above_coords.append(y_above_coord)
                        break
            median_y_below_coord = np.median(np.array(sorted(y_below_coords))).astype(int)
            median_y_above_coord = np.median(np.array(sorted(y_above_coords))).astype(int)
            self._y_coords = [median_y_above_coord + self._coord[1], median_y_below_coord + self._coord[1]]
        return self._y_coords

    @property
    def x_coords(self) -> tuple:
        if self._x_coords is None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
            erode_img = cv2.erode(self.binary_img.copy(), kernel=kernel, iterations=1)
            center_y = int(self.box_height / 2)
            center_line = erode_img[center_y:center_y + 1, 0:self.box_width].flatten()
            start_point = end_point = 0
            for i in range(0, self.box_width - 1):
                if center_line[i] == 0 and center_line[i + 1] == 0:
                    start_point = i
                    break
            for j in range(1, self.box_width):
                if center_line[-j] == 0 and center_line[-j - 1] == 0:
                    end_point = self.box_width - j
                    break
            self._x_coords = (start_point + self._coord[0], end_point + self._coord[0])
        return self._x_coords
    
    @property
    def character_height(self):
        if self._character_height is None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1))
            erode_img = cv2.erode(self.binary_img.copy(), kernel=kernel, iterations=3)
            above_point = below_point = None
            for i1 in range(self.box_height):
                for j1 in range(int(self.box_width / 2)):
                    if 255 not in erode_img[i1, j1:j1 + int(self.box_width / 2) + 1]:
                        above_point = i1
                        break
                if above_point is not None:
                    break
            for i2 in range(self.box_height):
                for j2 in range(int(self.box_width / 2)):
                    if 255 not in erode_img[-i2 - 1, j2:j2 + int(self.box_width / 2) + 1]:
                        below_point = self.box_height - i2 - 1
                        break
                if below_point is not None:
                    break
            if below_point is None or above_point is None:
                self._character_height = 0
            else:
                self._character_height = round((below_point - above_point), 1)
        return self._character_height
