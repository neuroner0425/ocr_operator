from paddleocr import PaddleOCR
import cv2
import os
import numpy as np
import json
from PIL import Image
from PIL import ImageDraw, ImageFont
from PIL import Image, ExifTags

lang = "korean" # 'ch', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka', 'latin', 'arabic', 'cyrillic', 'devanagari'

isPreprocessing = 1 # 0: 원본, 1: 전처리

file_name = "A" # 파일명
file_extension = ".jpeg" # 확장자

font_path = "C:/NanumGothic/NanumGothic.ttf" # 폰트 경로

# ----------------------------------------------

ocr = PaddleOCR(lang=lang)

img_path = f"{file_name}{file_extension}"

def rotate_image_based_on_exif(image_path):
    img = Image.open(image_path)

    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        
        exif = dict(img._getexif().items())

        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)

    except (AttributeError, KeyError, IndexError):
        pass

    return img

rotated_img = rotate_image_based_on_exif(img_path)
rotated_img_path = "rotated_" + img_path
rotated_img.save(rotated_img_path)

image = cv2.imread(rotated_img_path)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equ = cv2.equalizeHist(gray)

ret, thresh = cv2.threshold(equ, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

preprocessed_img_path = f"{file_name}_preprocessed{file_extension}"
cv2.imwrite(preprocessed_img_path, thresh)

if isPreprocessing:
    doPath = preprocessed_img_path
else:
    doPath = img_path

result = ocr.ocr(doPath, cls=True)

json_data = []

print(result)

for line in result[0]:
    text, confidence = line[1]
    box = line[0]
    if confidence > 0:
        detection = {
            "text": text,
            "confidence": confidence,
            "box": box
        }
        json_data.append(detection)
json_data_sorted = sorted(json_data, key=lambda x: x['confidence'], reverse=True)

print("-" * 50)
print(json_data)
print("-" * 50)

def draw_detections_and_texts(image_path, detections, output_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    if image.mode == 'RGB':
        outline_color = (255, 0, 0)
        stroke_color = (0, 0, 0)
    else:
        outline_color = 255
        stroke_color = 0 

    
    font_size = 18
    font = ImageFont.truetype(font_path, font_size)

    for detection in detections:
        text = detection["text"]
        confidence = detection["confidence"]
        box = detection["box"]

        box = np.array(box).astype(np.int32).reshape(-1, 2)
        min_x, min_y = np.min(box, axis=0)
        max_x, max_y = np.max(box, axis=0)

        top_left = (min_x, min_y)
        bottom_right = (max_x, max_y)

        draw.rectangle([top_left, bottom_right], outline=outline_color, width=2)
        text_position = (bottom_right[0] + 10, top_left[1])

        draw.text(text_position, text, font=font, fill=outline_color, stroke_width=1, stroke_fill=stroke_color)

    image.save(output_path)

img_path_out = f"{file_name}-{lang}({isPreprocessing})-out{file_extension}"

draw_detections_and_texts(rotated_img_path, json_data, img_path_out)

with open(f'ocr_result-{file_name}({isPreprocessing}).json', 'w', encoding='utf-8') as json_file:
    json.dump(json_data, json_file, ensure_ascii=False, indent=4)

for detection in json_data_sorted:
    text = detection["text"]
    confidence = detection["confidence"]
    box = detection["box"]
    print(text + " : " + str(confidence))