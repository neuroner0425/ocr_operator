import easyocr
import cv2
import json
import os
import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont

reader = easyocr.Reader(['en','ko'])

file_name = "ocr_test-2"
file_extension = ".png"
img_path = f"{file_name}{file_extension}"

image = cv2.imread(img_path)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equ = cv2.equalizeHist(gray)

ret, thresh = cv2.threshold(equ, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

preprocessed_img_path = f"{file_name}_preprocessed{file_extension}"
cv2.imwrite(preprocessed_img_path, thresh)



result = reader.readtext(img_path)

json_data = []

for (box, text, confidence) in result:
    if confidence > 0:
        box = [[int(point) for point in pair] for pair in box]
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

    font_path = "path_to_your_font/NanumGothic.ttf"
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

img_path_out = f"{file_name}-out{file_extension}"

draw_detections_and_texts(img_path, json_data, img_path_out)

with open('ocr_result.json', 'w', encoding='utf-8') as json_file:
    json.dump(json_data, json_file, ensure_ascii=False, indent=4)

for detection in json_data_sorted:
    text = detection["text"]
    confidence = detection["confidence"]
    box = detection["box"]
    print(text + " : " + str(confidence))