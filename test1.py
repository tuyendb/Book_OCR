import time
import numpy as np
import json
from objects.paragraph import Paragraph
from objects.book import Book
from objects.page import Page
from config import pdf_path
import matplotlib.pyplot as plt


class ExtractDataFromPDF:

    def __init__(self):
        self.book = Book
        self.page = Page
        self.paragraph = Paragraph

    @property
    def extract_data_to_string(self):
        st = time.time()
        book = self.book(pdf_path, 50, 50).pages
        paragrs = self.page(book[0]).get_paragraphs
        text = ""
        for key in list(paragrs.keys()):
            paragr = paragrs[key]["paragraph"]
            text += self.paragraph(paragr).content + '\n' + str(self.paragraph(paragr).is_title) + '.'*20 + '\n'
        total_time = time.time() - st
        return text, total_time

    @property
    def extract_features(self):
        book = self.book(pdf_path, None, None).pages
        paragraphs = []
        paragraph_id1 = []
        paragraph_id2 = []
        para_id = 0
        x_begin = []
        height = []
        thickness = []
        for page_id, page in enumerate(book):
            try:
                paragrs = self.page(page).get_paragraphs
                for key in list(paragrs.keys()):
                    paragr = paragrs[key]["paragraph"]
                    paragraphs.append(self.paragraph(paragr).content)
                    paragraph_id1.append(para_id)
                    paragraph_id2.append([page_id, key])
                    height.append(self.paragraph(paragr).parag_features[0])
                    thickness.append(self.paragraph(paragr).parag_features[1])
                    x_begin.append(self.paragraph(paragr).parag_features[2])
                    para_id += 1
            except IndexError:
                continue
        return paragraph_id1, paragraph_id2, height, thickness, x_begin, paragraphs


def main():
    extract = ExtractDataFromPDF()
    features = extract.extract_features
    features_data = {}
    for id in features[0]:
        features_data[id] = {}
        features_data[id]["para_id"] = features[1][id]
        features_data[id]["height"] = features[2][id]
        features_data[id]["thickness"] = features[3][id]
        features_data[id]["x_begin"] = features[4][id]
        features_data[id]["paragraph"] = features[5][id]
    json.dump(features_data, open('data6.json', 'w', encoding='utf-8'), ensure_ascii=False)


if __name__ == "__main__":
    main()
