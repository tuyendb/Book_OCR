import string


class Paragraph:

    def __init__(self, paragraph: list):
        self._paragraph = paragraph
        self._content = None
        self._paragr_features = None

    @property
    def content(self):
        if self._content is None:
            self._content = ""
            remove = ["BÀI", "Bài", "BAI", "Bai", "bài", "bai"]
            for text_line in self._paragraph:
                text = text_line.text
                if text.__len__() >= 8:
                    for rm in remove:
                        if rm in text[:3]:
                            index = text[4:].index(' ')
                            text = text.replace(text[0:index+5], '')
                self._content += text + " "
        return self._content

    @property
    def is_unnecessary(self) -> bool:
        unnecessary = ["Hình", "Hinh"]
        unncsr = False
        if self.content.__len__() >= 8:
            for unnec2 in unnecessary:
                if unnec2 in self.content and self.content[5].isdecimal():
                    unncsr = True
                    break
        return unncsr

    @property
    def is_title(self) -> bool:
        title = False
        lower_list = list(string.ascii_lowercase)
        upper_list = list(string.ascii_uppercase)
        double_lower_list = [2*ch for ch in lower_list]
        tripple_lower_list = [3*ch for ch in lower_list]
        double_upper_list = [2*ch for ch in upper_list]
        tripple_upper_list = [3*ch for ch in upper_list]
        check_title1 = lower_list + upper_list + double_upper_list + double_lower_list \
                       + tripple_upper_list + tripple_lower_list
        for num in range(1, 100):
            check_title1.append(str(num))
        check_title2 = ['/', ')', '.', ':', '-', '?', ' ']
        if self.content.isupper():
            title = True
        elif self.content[:5].isupper():
            title = True
        else:
            for bg_ch1 in check_title1:
                for bg_ch2 in check_title2:
                    begin_crts = bg_ch1 + bg_ch2
                    bc_len = begin_crts.__len__()
                    if self.content.__len__() > (bc_len + 1):
                        if ((begin_crts in self.content[:bc_len])
                                and (self.content[bc_len].isupper()
                                     or self.content[bc_len + 1].isupper())):
                            title = True
                            break
        return title

    @property
    def parag_features(self):
        if self._paragr_features is None:
            paragr_height = paragr_thickness = 0
            max_length = 0
            for text_line in self._paragraph:
                max_length = max((text_line.x_coords[1] - text_line.x_coords[0]), max_length)
            for text_line in self._paragraph:
                if text_line.x_coords[1] - text_line.x_coords[0] == max_length:
                    paragr_height = text_line.character_height
                    paragr_thickness = text_line.character_thickness
                    break
            paragr_x_begin = self._paragraph[0].x_coords[0]
            self._paragr_features = (paragr_height, paragr_thickness, paragr_x_begin)
        return self._paragr_features
