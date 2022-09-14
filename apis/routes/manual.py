from apis.routes.base_route import BaseRoute
from objects.data_request import GetDataFromPDF
from pipeline import ExtractDataFromPDF
from fastapi import File, Form


class ManualRoute(BaseRoute):
    def __init__(self):
        super(ManualRoute, self).__init__(prefix="/getdata")
        self.pipeline = ExtractDataFromPDF()

    def create_routes(self):
        router = self.router

        @router.get("/")
        async def get_data(pdf_file: bytes = File(), first_page=Form(...), last_page=Form(...)):
            data = GetDataFromPDF(pdf_file, int(first_page), int(last_page))
            data = self.pipeline(data)
            output = data.data
            return {"Content": output[0],
                    "Time": output[1]}
