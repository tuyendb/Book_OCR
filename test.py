import time
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from objects.paragraph import Paragraph
from objects.book import Book
from objects.page import Page
from config import pdf_path


class ExtractDataFromPDF:

    def __init__(self):
        self.book = Book
        self.page = Page
        self.paragraph = Paragraph

    @property
    def extract_data_to_string(self):
        st = time.time()
        book = self.book(pdf_path, 27, 27).pages
        paragrs = self.page(book[0]).get_paragraphs
        text = ""
        for key in list(paragrs.keys()):
            paragr = paragrs[key]["paragraph"]
            text += self.paragraph(paragr).content + '\n' + str(self.paragraph(paragr).is_title) + '.'*20 + '\n'
        total_time = time.time() - st
        return text, total_time

    @property
    def extract_features(self):
        book = self.book(pdf_path, 27, 27).pages
        height = []
        thickness = []
        x_begin = []
        paragraphs_id = []
        paragraphs = []
        para_id = 0
        # page_id = -1
        # parag_id = []
        for page in book:
            # page_id += 1
            try:
                paragrs = self.page(page).get_paragraphs
                for key in list(paragrs.keys()):
                    paragr = paragrs[key]["paragraph"]
                    height.append(self.paragraph(paragr).parag_features[0])
                    thickness.append(self.paragraph(paragr).parag_features[1])
                    paragraphs_id.append(para_id)
                    x_begin.append(self.paragraph(paragr).parag_features[2])
                    paragraphs.append(self.paragraph(paragr).content)
                    para_id += 1
                    # parag_id.append([page_id, key])
            except IndexError:
                continue
        return height, thickness, x_begin, paragraphs_id#, paragraphs, parag_id


def main():
    extract = ExtractDataFromPDF()
    features = extract.extract_features
    features_data = {}
    for id in features[3]:
        features_data[id] = {}
        features_data[id]["thickness"] = features[1][id]
        features_data[id]["height"] = features[0][id]
        features_data[id]["x_begin"] = features[2][id]
        # features_data[id]["paragraph"] = features[4][id]
        # features_data[id]["para_id"] = features[5][id]
    json.dump(features_data, open('data8.json', 'w', encoding='utf-8'), ensure_ascii=False)
    # with open('data4.json') as jsonfile:
    #     data = json.load(jsonfile)
    #     paragr_heights = []
    #     paragr_thicknesses = []
    #     paragr_x_begin = []
    #     for id in data.keys():
    #         paragr_heights.append(data[id]['height'])
    #         paragr_thicknesses.append(data[id]['thickness'])
    #         paragr_x_begin.append(data[id]['x_begin'])
    # print(paragr_heights.__len__())
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x = np.array(paragr_x_begin)
    # y = np.array(paragr_thicknesses)
    # z = np.array(paragr_heights)
    # ax.set_xlabel('X_begin')
    # ax.set_ylabel('Thickness')
    # ax.set_zlabel('Height')
    # img = ax.scatter(x, y, z)
    # fig.colorbar(img)
    # plt.show()


if __name__ == "__main__":
    main()
