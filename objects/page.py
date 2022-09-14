from utils import ImgDetector
from init_models import models_init
from objects.text_line import TextLine
import numpy as np


text_detector = models_init()[0]


class Page:

    def __init__(self, page_img):
        self._page_img = page_img
        self._lines = None
        self._line_dis = None

    @property
    def page_img(self):
        return self._page_img

    @property
    def lines(self):
        if self._lines is None:
            self._lines = ImgDetector(self._page_img.copy(), text_detector).crop2rec
        return self._lines

    @property
    def line_distance(self):
        if self._line_dis is None:
            line_distances = []
            for line_id in range(self.lines[0].__len__() - 1):
                y_above = TextLine(self.lines[0][line_id], self.lines[1][line_id]).y_coords[1]
                y_below = TextLine(self.lines[0][line_id+1], self.lines[1][line_id+1]).y_coords[0]
                dis = y_below - y_above
                if dis > 0:
                    line_distances.append(dis)
                self._line_dis = np.median(np.array(sorted(line_distances)))
        return self._line_dis

    def paragr_init(self, paragraphs, text_line, paragr_id) -> dict:
        paragraphs[paragr_id] = {}
        paragraphs[paragr_id]["paragraph"] = [text_line]
        paragraphs[paragr_id]["std_condition"] = [text_line.x_coords, text_line.character_thickness, text_line.y_coords,
                                                  self.line_distance, True]
        return paragraphs

    @staticmethod
    def check_belong_to_pgr(text_line, paragraphs: dict):
        para_ids = list(paragraphs.keys())
        para_ids.reverse()
        belong = None
        paragr_id = None
        for para_id in para_ids:
            std_text_line = paragraphs[para_id]["paragraph"][0]
            std_x_coords, std_chara_thickness, std_y_coords = paragraphs[para_id]["std_condition"][:3]
            std_y_line_dis, check_para_init = paragraphs[para_id]["std_condition"][3:]
            cdt1 = ((abs(std_x_coords[0] - text_line.x_coords[0]) <= 7*std_chara_thickness)
                    or (abs(text_line.x_coords[1] - std_x_coords[1]) <= std_chara_thickness)) \
                and (text_line.y_coords[0] > std_y_coords[1])
            if check_para_init:
                if std_text_line.check_not_subtitle_list:
                    if text_line.check_first_upper:
                        cdt2 = ((text_line.y_coords[0] - std_y_coords[1]) <= 1.5*std_y_line_dis) \
                               and text_line.check_not_subtitle_list
                    else:
                        cdt2 = text_line.check_not_subtitle_list and \
                           ((text_line.y_coords[0] - std_y_coords[1]) <= 1.5*std_y_line_dis)
                else:
                    if text_line.check_first_upper:
                        cdt2 = ((text_line.y_coords[0] - std_y_coords[1]) <= 1.3*std_y_line_dis) and \
                               text_line.check_not_subtitle_list and \
                               ((text_line.x_coords[1] - std_text_line.x_coords[1]) <= 2 * std_chara_thickness)
                    else:
                        cdt2 = text_line.check_not_subtitle_list and \
                               ((text_line.y_coords[0] - std_y_coords[1]) <= 1.5*std_y_line_dis) and \
                               ((text_line.x_coords[1] - std_text_line.x_coords[1]) <= 2 * std_chara_thickness)
            else:
                if text_line.check_first_upper:
                    if text_line.x_coords[0] >= (std_x_coords[0] + 2*std_chara_thickness):
                        cdt2 = False
                    else:
                        cdt2 = (text_line.y_coords[0]-std_y_coords[1]) <= 1.25*std_y_line_dis
                else:
                    cdt2 = text_line.check_not_subtitle_list and \
                           ((text_line.y_coords[0] - std_y_coords[1]) <= 1.5*std_y_line_dis)
            if cdt1 and cdt2:
                belong = True
                paragr_id = para_id
                break
            else:
                belong = False
                paragr_id = para_ids[0]
        return belong, paragr_id

    @property
    def get_paragraphs(self) -> dict:
        paragraphs = {}
        paragr_id = 0
        for text_line_id in range(self.lines[0].__len__()):
            text_line = TextLine(self.lines[0][text_line_id], self.lines[1][text_line_id])
            if text_line.check_unnecessary:
                continue
            if paragraphs.__len__() == 0:
                paragraphs = self.paragr_init(paragraphs, text_line, paragr_id)
            else:

                belong, paragr_id = self.check_belong_to_pgr(text_line, paragraphs)

                if belong:
                    paragraphs[paragr_id]["paragraph"].append(text_line)
                    paragraphs[paragr_id]["std_condition"][0] = text_line.x_coords
                    paragraphs[paragr_id]["std_condition"][1] = 0.5 * (text_line.character_thickness
                                                                       + paragraphs[paragr_id]["std_condition"][1]
                                                                       )
                    paragraphs[paragr_id]["std_condition"][3] = \
                        text_line.y_coords[0] - paragraphs[paragr_id]["std_condition"][2][1]
                    paragraphs[paragr_id]["std_condition"][2] = text_line.y_coords
                    paragraphs[paragr_id]["std_condition"][4] = False
                else:
                    paragraphs = self.paragr_init(paragraphs, text_line, paragr_id + 1)
        para_ids = list(paragraphs.keys())
        for para_id in para_ids:
            del paragraphs[para_id]["std_condition"]
        return paragraphs
