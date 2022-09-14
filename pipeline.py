from objects.data_request import GetDataFromPDF
from objects.paragraph import Paragraph
from objects.book import Book
from objects.page import Page
import time


class ExtractDataFromPDF:

    def __init__(self):
        self.book = Book
        self.page = Page
        self.paragraph = Paragraph

    def extract_data_to_string(self, pdf_file, first_page, last_page):
        st = time.time()
        book = self.book(pdf_file, first_page, last_page).pages
        paragrs = self.page(book[0]).get_paragraphs
        text = ""
        for key in list(paragrs.keys()):
            paragr = paragrs[key]["paragraph"]
            text += self.paragraph(paragr).content + '\n'
        total_time = time.time() - st
        return text, total_time

    # def structure_extracted_data(self, pdf_file, first_page, last_page):
    #     st = time.time()
    #     book = self.book(pdf_file, first_page, last_page).pages
    #     for page in book:
    #         paragrs = self.page(page).get_paragraphs
    #         for key in list(paragrs.keys()):
    #             paragr = paragrs[key]["paragraph"]

    def __call__(self, data: GetDataFromPDF) -> GetDataFromPDF:
        data_extract = self.extract_data_to_string(data.pdf_file, data.first_page, data.last_page)
        data.data = [data_extract[0], data_extract[1]]
        return data
