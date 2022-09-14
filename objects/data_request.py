class GetDataFromPDF:
    data = None

    def __init__(self, pdf_file, first_page, last_page):
        self._pdf_file = pdf_file
        self._first_page = first_page
        self._last_page = last_page

    @property
    def pdf_file(self):
        return self._pdf_file

    @property
    def first_page(self):
        return self._first_page

    @property
    def last_page(self):
        return self._last_page
