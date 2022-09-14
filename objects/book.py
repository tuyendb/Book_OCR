from utils import pdf2jpg


class Book:

    def __init__(self, pdf_path, first_page, last_page):
        self._pdf_path = pdf_path
        self._first_page = first_page
        self._last_page = last_page

    @property
    def pages(self):
        return pdf2jpg(self._pdf_path, self._first_page, self._last_page)
