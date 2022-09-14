import cv2


def main():
    img_file_path = '../book/results/1.jpg'
    img = cv2.imread(img_file_path)
    gray_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(gray_img.copy(), 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    erode_img = cv2.erode(thresh_img.copy(), kernel=kernel, iterations=3)
    height, width = erode_img.shape[:2]
    new_erode_img = erode_img[0:height, 0:width]  # int(0.1 * width):int(0.9 * width)
    new_height, new_width = new_erode_img.shape[:2]
    above_point = below_point = None
    for i1 in range(new_height):
        for j1 in range(new_width - 1):
            start_point = end_point = -1
            if new_erode_img[i1, j1] == 0 and new_erode_img[i1, j1 + 1] == 0:
                start_point = j1
            if j1 > start_point > -1:
                if new_erode_img[i1, j1] == 0 and new_erode_img[i1, j1 + 1] == 255:
                    end_point = j1
                elif new_erode_img[i1, j1] == 0 and new_erode_img[i1, j1 + 1] == 0 and j1 + 1 == (new_width - 1):
                    end_point = j1 + 1
            if end_point > start_point > -1 and (end_point - start_point) >= new_width / 2:
                above_point = i1
                break
        if above_point is not None:
            break
    for i2 in range(new_height):
        for j2 in range(new_width - 1):
            start_point = end_point = -1
            if new_erode_img[-i2 - 1, j2] == 0 and new_erode_img[-i2 - 1, j2 + 1] == 0:
                start_point = j2
            if j2 > start_point > -1:
                if new_erode_img[-i2 - 1, j2] == 0 and new_erode_img[-i2 - 1, j2 + 1] == 255:
                    end_point = j2
                elif new_erode_img[-i2 - 1, j2] == 0 and new_erode_img[-i2 - 1, j2 + 1] == 0 and j2 + 1 == (
                        new_width - 1):
                    end_point = j2 + 1
            if end_point > start_point > -1 and (end_point - start_point) >= new_width / 2:
                below_point = new_height - i2 - 1
                break
        if below_point is not None:
            break
    min_character_height = below_point - above_point
    print(min_character_height)
    print(above_point, below_point, new_height)
    cv2.imshow('erode', new_erode_img)
    cv2.waitKey()


if __name__ == "__main__":
    main()
