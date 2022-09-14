from objects.paragraph import Paragraph
from objects.book import Book
from objects.page import Page
from config import pdf_path
import time
import numpy as np
import matplotlib.pyplot as plt


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
        para_id = 0
        for page in book:
            paragrs = self.page(page).get_paragraphs
            for key in list(paragrs.keys()):
                paragr = paragrs[key]["paragraph"]
                height.append(self.paragraph(paragr).parag_features[0])
                thickness.append(self.paragraph(paragr).parag_features[1])
                paragraphs_id.append(para_id)
                x_begin.append(self.paragraph(paragr).x_begin_coord)
                para_id += 1
        return height, thickness, x_begin, paragraphs_id


def main():
    extract = ExtractDataFromPDF()
    features = extract.extract_features
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.array(features[0])
    y = np.array(features[1])
    z = np.array(features[2])
    c = np.array(features[3])
    img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
    fig.colorbar(img)
    plt.show()


if __name__ == "__main__":
    main()