import time
start = time.time()
import cv2
import os
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont, ExifTags
import math
import re
from datetime import datetime

filepath = "A.jpeg"

file_name, file_extension = os.path.splitext(filepath)
if not file_extension.startswith('.'):
    file_extension = '.' + file_extension

result_en0 = []
with open(f'./result/ocr_result-{file_name}-en(0).json', 'r') as file:
    result_en0 = json.load(file)
result_en1 = []
with open(f'./result/ocr_result-{file_name}-en(1).json', 'r') as file:
    result_en1 = json.load(file)
result_ko0 = []
with open(f'./result/ocr_result-{file_name}-korean(0).json', 'r') as file:
    result_ko0 = json.load(file)
result_ko1 = []
with open(f'./result/ocr_result-{file_name}-korean(1).json', 'r') as file:
    result_ko1 = json.load(file)
result_enko0 = []
with open(f'./result/ocr_result-{file_name}-enko(0).json', 'r') as file:
    result_enko0 = json.load(file)
result_enko1 = []
with open(f'./result/ocr_result-{file_name}-enko(1).json', 'r') as file:
    result_enko1 = json.load(file)

results = [result_en0, result_en1, result_ko0, result_ko1, result_enko0, result_enko1]

# 모든 가능한 날짜 패턴을 생성
date_patterns = [
    r'\d{4}\.\d{1,2}\.\d{1,2}',             # yyyy.mm.dd
    r'\d{2}\.\d{1,2}\.\d{1,2}',             # yy.mm.dd
    r'\d{1,2}\.\d{1,2}\.\d{4}',             # dd.mm.yyyy
    r'\d{1,2}\.\d{1,2}',                    # mm.dd (현재 연도 가정)
    r'\d{2,4}\.\d{1,2}\.\d{1,2}\(\w*\)',    # yyyy.mm.dd(요일)
    r'\d{1,2}\.\d{1,2}\(\w*\)',             # mm.dd(요일) (현재 연도 가정)
    '\d{1,2}\.\d{1,2}\(?\w*?\)?',           # mm.dd(요일) (현재 연도 가정)
    '\\d{1,2}\\.Mar(?:ch)?\\.\\d{4}',       # dd.Mar.yyyy
    'Aug(?:ust)?-\\d{1,2}-\\d{2}',          # Aug-dd-yy
    'Feb(?:ruary)?/\\d{1,2}/\\d{2}',        # Feb/dd/yy
    '\\d{4}/Dec(?:ember)?/\\d{1,2}',        # yyyy/Dec/dd
    '\\d{4}-May-\\d{1,2}',                  # yyyy-May-dd
    '\\d{4}/Jul(?:y)?/\\d{1,2}',            # yyyy/Jul/dd
    '\\d{1,2}/Sep(?:tember)?/\\d{4}',       # dd/Sep/yyyy
    '\\d{4}-\\d{1,2}-\\d{1,2}',             # yyyy-mm-dd
    '\\d{4}\\sSep(?:tember)?\\s\\d{1,2}',   # yyyy Sep dd
    '\\d{4}\\.Dec(?:ember)?\\.\\d{1,2}',    # yyyy.Dec.dd
    '\\d{1,2}/Jan(?:uary)?/\\d{2}',         # dd/Jan/yy
    '\\d{2}\\.Jul(?:y)?\\.\\d{1,2}',        # yy.Jul.dd
    '\\d{2}/Aug(?:ust)?/\\d{1,2}',          # yy/Aug/dd
    '\\d{4}\\.Apr(?:il)?\\.\\d{1,2}',       # yyyy.Apr.dd
    '\\d{1,2}/Feb(?:ruary)?/\\d{4}',        # dd/Feb/yyyy
    '\\d{4}\\s\\d{1,2}\\s\\d{1,2}',         # yyyy mm dd
    '\\d{2}/Feb(?:ruary)?/\\d{1,2}',        # yy/Feb/dd
    '\\d{1,2}\\.Apr(?:il)?\\.\\d{2}',       # dd.Apr.yy
    '\\d{1,2}\\.Jul(?:y)?\\.\\d{2}',        # dd.Jul.yy
    'Dec(?:ember)?\\s\\d{1,2}\\s\\d{2}',    # Dec dd yy
    '\\d{1,2}/Oct(?:ober)?/\\d{4}',         # dd/Oct/yyyy
    '\\d{1,2}/Dec(?:ember)?/\\d{4}',        # dd/Dec/yyyy
    '\\d{1,2}-Aug(?:ust)?-\\d{4}',          # dd-Aug-yyyy
    '\\d{1,2}\\sJul(?:y)?\\s\\d{2}',        # dd Jul yy
    '\\d{1,2}\\sJul(?:y)?\\s\\d{4}',        # dd Jul yyyy
    '\\d{1,2}-Feb(?:ruary)?-\\d{2}',        # dd-Feb-yy
    '\\d{1,2}-Apr(?:il)?-\\d{4}',           # dd-Apr-yyyy
    '\\d{2}\\sMay\\s\\d{1,2}',              # yy May dd
    '\\d{1,2}-Jul(?:y)?-\\d{2}',            # dd-Jul-yy
    '\\d{4}\\.Mar(?:ch)?\\.\\d{1,2}',       # yyyy.Mar.dd
    '\\d{2}\\.Feb(?:ruary)?\\.\\d{1,2}',    # yy.Feb.dd
    '\\d{1,2}/Aug(?:ust)?/\\d{2}',          # dd/Aug/yy
    '\\d{1,2}-Sep(?:tember)?-\\d{4}',       # dd-Sep-yyyy
    '\\d{4}/Sep(?:tember)?/\\d{1,2}',       # yyyy/Sep/dd
    '\\d{1,2}\\sDec(?:ember)?\\s\\d{2}',    # dd Dec yy
    '\\d{4}/\\d{1,2}/\\d{1,2}',             # yyyy/mm/dd
    '\\d{4}-Aug(?:ust)?-\\d{1,2}',          # yyyy-Aug-dd
    '\\d{1,2}\\sFeb(?:ruary)?\\s\\d{2}',    # dd Feb yy
    '\\d{2}-May-\\d{1,2}',                  # yy-May-dd
    '\\d{1,2}-Nov(?:ember)?-\\d{2}',        # dd-Nov-yy
    '\\d{4}\\sJun(?:e)?\\s\\d{1,2}',        # yyyy Jun dd
    '\\d{1,2}\\.\\d{1,2}\\.\\d{4}',         # dd.mm.yyyy
    '\\d{2}/Jan(?:uary)?/\\d{1,2}',         # yy/Jan/dd
    '\\d{2}-Jan(?:uary)?-\\d{1,2}',         # yy-Jan-dd
    '\\d{2}\\sApr(?:il)?\\s\\d{1,2}',       # yy Apr dd
    'Apr(?:il)?\\.\\d{1,2}\\.\\d{2}',       # Apr.dd.yy
    '\\d{1,2}\\sNov(?:ember)?\\s\\d{2}',    # dd Nov yy
    '\\d{2}\\.Oct(?:ober)?\\.\\d{1,2}',     # yy.Oct.dd
    '\\d{4}/Jan(?:uary)?/\\d{1,2}',         # yyyy/Jan/dd
    '\\d{1,2}/\\d{1,2}/\\d{4}',             # dd/mm/yyyy
    '\\d{1,2}-Feb(?:ruary)?-\\d{4}',        # dd-Feb-yyyy
    '\\d{1,2}\\sMay\\s\\d{4}',              # dd May yyyy
    '\\d{2}\\s\\d{1,2}\\s\\d{1,2}',         # yy mm dd
    '\\d{4}\\.Feb(?:ruary)?\\.\\d{1,2}',    # yyyy.Feb.dd
    '\\d{1,2}-Dec(?:ember)?-\\d{4}',        # dd-Dec-yyyy
    '\\d{4}\\.Jun(?:e)?\\.\\d{1,2}',        # yyyy.Jun.dd
    '\\d{1,2}-Sep(?:tember)?-\\d{2}',       # dd-Sep-yy
    '\\d{1,2}\\sDec(?:ember)?\\s\\d{4}',    # dd Dec yyyy
    '\\d{2}/Dec(?:ember)?/\\d{1,2}',        # yy/Dec/dd
    '\\d{2}\\.Nov(?:ember)?\\.\\d{1,2}',    # yy.Nov.dd
    'Jun(?:e)?\\s\\d{1,2}\\s\\d{2}',        # Jun dd yy
    '\\d{1,2}-Oct(?:ober)?-\\d{4}',         # dd-Oct-yyyy
    '\\d{2}\\.Apr(?:il)?\\.\\d{1,2}',       # yy.Apr.dd
    '\\d{4}/Oct(?:ober)?/\\d{1,2}',         # yyyy/Oct/dd
    'Oct(?:ober)?\\.\\d{1,2}\\.\\d{2}',     # Oct.dd.yy
    '\\d{4}/Feb(?:ruary)?/\\d{1,2}',        # yyyy/Feb/dd
    '\\d{4}\\.Nov(?:ember)?\\.\\d{1,2}',    # yyyy.Nov.dd
    '\\d{2}\\sMar(?:ch)?\\s\\d{1,2}',       # yy Mar dd
    '\\d{1,2}/Jan(?:uary)?/\\d{4}',         # dd/Jan/yyyy
    '\\d{1,2}\\sJan(?:uary)?\\s\\d{4}',     # dd Jan yyyy
    '\\d{4}\\.Jul(?:y)?\\.\\d{1,2}',        # yyyy.Jul.dd
    '\\d{2}\\.Jun(?:e)?\\.\\d{1,2}',        # yy.Jun.dd 
    '\\d{2}-Dec(?:ember)?-\\d{1,2}',        # yy-Dec-dd
    '\\d{4}\\sMar(?:ch)?\\s\\d{1,2}',       # yyyy Mar dd
    'Apr(?:il)?\\s\\d{1,2}\\s\\d{2}',       # Apr dd yy
    '\\d{1,2}\\sJan(?:uary)?\\s\\d{2}',     # dd Jan yy
    '\\d{2}-Jun(?:e)?-\\d{1,2}',            # yy-Jun-dd
    'Sep(?:tember)?\\s\\d{1,2}\\s\\d{2}',   # Sep dd yy
    'Nov(?:ember)?/\\d{1,2}/\\d{2}',        # Nov/dd/yy
    '\\d{1,2}-Mar(?:ch)?-\\d{4}',           # dd-Mar-yyyy
    '\\d{4}\\sOct(?:ober)?\\s\\d{1,2}',     # yyyy Oct dd
    '\\d{1,2}\\sSep(?:tember)?\\s\\d{4}',   # dd Sep yyyy
    'Jan(?:uary)?\\.\\d{1,2}\\.\\d{2}',     # Jan.dd.yy
    '\\d{1,2}\\.Jan(?:uary)?\\.\\d{4}',     # dd.Jan.yyyy
    '\\d{2}-Sep(?:tember)?-\\d{1,2}',       # yy-Sep-dd
    '\\d{1,2}\\.Sep(?:tember)?\\.\\d{4}', 
    '\\d{1,2}/Jun(?:e)?/\\d{2}', 
    '\\d{1,2}\\.Aug(?:ust)?\\.\\d{4}', 
    '\\d{1,2}\\sNov(?:ember)?\\s\\d{4}', 
    '\\d{4}-Sep(?:tember)?-\\d{1,2}', 
    'Jan(?:uary)?-\\d{1,2}-\\d{2}', 
    '\\d{1,2}\\.\\d{1,2}\\.\\d{2}', 
    '\\d{2}\\sFeb(?:ruary)?\\s\\d{1,2}', 
    '\\d{4}-Mar(?:ch)?-\\d{1,2}', 
    '\\d{1,2}\\sAug(?:ust)?\\s\\d{2}', 
    '\\d{1,2}/Apr(?:il)?/\\d{4}', 
    '\\d{2}\\.May\\.\\d{1,2}', 
    'Sep(?:tember)?-\\d{1,2}-\\d{2}', 
    '\\d{1,2}/Oct(?:ober)?/\\d{2}', 
    '\\d{1,2}-Jun(?:e)?-\\d{2}', 
    '\\d{2}\\sJan(?:uary)?\\s\\d{1,2}', 
    '\\d{1,2}/Dec(?:ember)?/\\d{2}', 
    '\\d{1,2}\\.Mar(?:ch)?\\.\\d{2}', 
    'Aug(?:ust)?/\\d{1,2}/\\d{2}', 
    'Jun(?:e)?/\\d{1,2}/\\d{2}', 
    'Jul(?:y)?\\.\\d{1,2}\\.\\d{2}', 
    '\\d{1,2}\\sApr(?:il)?\\s\\d{4}', 
    '\\d{1,2}/Mar(?:ch)?/\\d{4}', 
    'Feb(?:ruary)?\\.\\d{1,2}\\.\\d{2}', 
    '\\d{4}/May/\\d{1,2}', 
    'Aug(?:ust)?\\.\\d{1,2}\\.\\d{2}', 
    '\\d{1,2}-\\d{1,2}-\\d{2}', 
    '\\d{2}-Mar(?:ch)?-\\d{1,2}', 
    '\\d{4}\\.Aug(?:ust)?\\.\\d{1,2}', 
    '\\d{1,2}\\.May\\.\\d{4}', 
    '\\d{2}\\sNov(?:ember)?\\s\\d{1,2}', 
    '\\d{1,2}\\sApr(?:il)?\\s\\d{2}', 
    '\\d{4}\\.Sep(?:tember)?\\.\\d{1,2}', 
    'Feb(?:ruary)?\\s\\d{1,2}\\s\\d{2}', 
    '\\d{4}\\sMay\\s\\d{1,2}', 
    '\\d{1,2}-Aug(?:ust)?-\\d{2}', 
    '\\d{1,2}\\sOct(?:ober)?\\s\\d{4}', 
    'Nov(?:ember)?\\s\\d{1,2}\\s\\d{2}', 
    'Dec(?:ember)?-\\d{1,2}-\\d{2}', 
    '\\d{4}\\.Jan(?:uary)?\\.\\d{1,2}', 
    '\\d{2}\\.Sep(?:tember)?\\.\\d{1,2}', 
    '\\d{1,2}-Mar(?:ch)?-\\d{2}', 
    'Jan(?:uary)?/\\d{1,2}/\\d{2}', 
    '\\d{2}\\sOct(?:ober)?\\s\\d{1,2}', 
    '\\d{1,2}\\.Jun(?:e)?\\.\\d{2}', 
    'May-\\d{1,2}-\\d{2}', 
    '\\d{2}\\sDec(?:ember)?\\s\\d{1,2}', 
    'Apr(?:il)?/\\d{1,2}/\\d{2}', 
    'Jul(?:y)?\\s\\d{1,2}\\s\\d{2}', 
    'Aug(?:ust)?\\s\\d{1,2}\\s\\d{2}', 
    '\\d{4}-Feb(?:ruary)?-\\d{1,2}', 
    '\\d{1,2}\\.Feb(?:ruary)?\\.\\d{2}', 
    '\\d{1,2}\\s\\d{1,2}\\s\\d{4}', 
    '\\d{1,2}\\.Dec(?:ember)?\\.\\d{4}', 
    '\\d{1,2}-Jan(?:uary)?-\\d{4}', 
    'Mar(?:ch)?/\\d{1,2}/\\d{2}', 
    'May\\.\\d{1,2}\\.\\d{2}', 
    '\\d{1,2}\\s\\d{1,2}\\s\\d{2}', 
    '\\d{1,2}/May/\\d{4}', 
    '\\d{1,2}\\sOct(?:ober)?\\s\\d{2}', 
    'Nov(?:ember)?\\.\\d{1,2}\\.\\d{2}', 
    '\\d{2}/Apr(?:il)?/\\d{1,2}', 
    '\\d{1,2}-Dec(?:ember)?-\\d{2}', 
    '\\d{2}/Jun(?:e)?/\\d{1,2}', 
    '\\d{1,2}\\sMar(?:ch)?\\s\\d{4}', 
    '\\d{1,2}\\.Sep(?:tember)?\\.\\d{2}', 
    '\\d{1,2}\\.Nov(?:ember)?\\.\\d{2}', 
    '\\d{1,2}-Jun(?:e)?-\\d{4}', 
    '\\d{1,2}\\sJun(?:e)?\\s\\d{4}', 
    '\\d{1,2}\\.Jun(?:e)?\\.\\d{4}', 
    '\\d{4}\\sAug(?:ust)?\\s\\d{1,2}', 
    '\\d{4}/Nov(?:ember)?/\\d{1,2}', 
    '\\d{4}-Jun(?:e)?-\\d{1,2}', 
    'Jun(?:e)?\\.\\d{1,2}\\.\\d{2}', 
    '\\d{4}\\.\\d{1,2}\\.\\d{1,2}', 
    '\\d{2}-\\d{1,2}-\\d{1,2}', 
    '\\d{4}-Jul(?:y)?-\\d{1,2}', 
    '\\d{1,2}-May-\\d{2}', 
    '\\d{4}-Apr(?:il)?-\\d{1,2}', 
    'Oct(?:ober)?\\s\\d{1,2}\\s\\d{2}', 
    '\\d{1,2}\\.Jan(?:uary)?\\.\\d{2}', 
    '\\d{1,2}\\sMar(?:ch)?\\s\\d{2}', 
    '\\d{1,2}/Feb(?:ruary)?/\\d{2}', 
    '\\d{2}\\.Jan(?:uary)?\\.\\d{1,2}', 
    'Jun(?:e)?-\\d{1,2}-\\d{2}', 
    'Oct(?:ober)?/\\d{1,2}/\\d{2}', 
    '\\d{1,2}/Nov(?:ember)?/\\d{2}', 
    '\\d{1,2}\\.Apr(?:il)?\\.\\d{4}', 
    '\\d{1,2}-Jan(?:uary)?-\\d{2}', 
    '\\d{2}/Mar(?:ch)?/\\d{1,2}', 
    '\\d{4}\\sFeb(?:ruary)?\\s\\d{1,2}', 
    '\\d{4}-Oct(?:ober)?-\\d{1,2}', 
    '\\d{4}-Nov(?:ember)?-\\d{1,2}', 
    '\\d{2}\\sSep(?:tember)?\\s\\d{1,2}', 
    '\\d{1,2}-May-\\d{4}', 
    '\\d{1,2}-Apr(?:il)?-\\d{2}', 
    '\\d{1,2}/Jul(?:y)?/\\d{4}', 
    '\\d{4}/Jun(?:e)?/\\d{1,2}', 
    '\\d{4}/Aug(?:ust)?/\\d{1,2}', 
    '\\d{2}-Jul(?:y)?-\\d{1,2}', 
    '\\d{2}\\.Dec(?:ember)?\\.\\d{1,2}', 
    '\\d{1,2}-Oct(?:ober)?-\\d{2}', 
    'May/\\d{1,2}/\\d{2}', 
    '\\d{1,2}\\sAug(?:ust)?\\s\\d{4}', 
    '\\d{1,2}/Nov(?:ember)?/\\d{4}', 
    'Jan(?:uary)?\\s\\d{1,2}\\s\\d{2}', 
    '\\d{1,2}/May/\\d{2}', 
    'May\\s\\d{1,2}\\s\\d{2}', 
    'Sep(?:tember)?\\.\\d{1,2}\\.\\d{2}', 
    '\\d{1,2}\\sMay\\s\\d{2}', 
    '\\d{1,2}/Sep(?:tember)?/\\d{2}', 
    '\\d{1,2}\\.Feb(?:ruary)?\\.\\d{4}', 
    '\\d{2}-Apr(?:il)?-\\d{1,2}', 
    '\\d{4}-Dec(?:ember)?-\\d{1,2}', 
    '\\d{2}/\\d{1,2}/\\d{1,2}', 
    '\\d{1,2}-\\d{1,2}-\\d{4}', 
    '\\d{4}\\sDec(?:ember)?\\s\\d{1,2}', 
    'Nov(?:ember)?-\\d{1,2}-\\d{2}', 
    '\\d{2}-Aug(?:ust)?-\\d{1,2}', 
    'Mar(?:ch)?\\s\\d{1,2}\\s\\d{2}', 
    '\\d{2}\\.Aug(?:ust)?\\.\\d{1,2}', 
    '\\d{1,2}\\.Oct(?:ober)?\\.\\d{2}', 
    '\\d{2}-Oct(?:ober)?-\\d{1,2}', 
    '\\d{2}-Feb(?:ruary)?-\\d{1,2}', 
    'Sep(?:tember)?/\\d{1,2}/\\d{2}', 
    '\\d{2}/Sep(?:tember)?/\\d{1,2}', 
    '\\d{1,2}/Jul(?:y)?/\\d{2}', 
    '\\d{1,2}\\.Nov(?:ember)?\\.\\d{4}', 
    'Mar(?:ch)?\\.\\d{1,2}\\.\\d{2}', 
    '\\d{4}-Jan(?:uary)?-\\d{1,2}', 
    '\\d{4}\\sJan(?:uary)?\\s\\d{1,2}', 
    '\\d{1,2}\\sJun(?:e)?\\s\\d{2}', 
    '\\d{2}\\.Mar(?:ch)?\\.\\d{1,2}', 
    '\\d{1,2}/Jun(?:e)?/\\d{4}', 
    'Oct(?:ober)?-\\d{1,2}-\\d{2}', 
    '\\d{1,2}\\sFeb(?:ruary)?\\s\\d{4}', 
    '\\d{4}\\sJul(?:y)?\\s\\d{1,2}', 
    '\\d{1,2}\\.Jul(?:y)?\\.\\d{4}', 
    '\\d{2}\\sJun(?:e)?\\s\\d{1,2}', 
    'Dec(?:ember)?/\\d{1,2}/\\d{2}', 
    '\\d{1,2}/\\d{1,2}/\\d{2}', 
    '\\d{4}\\sNov(?:ember)?\\s\\d{1,2}', 
    '\\d{4}\\.May\\.\\d{1,2}', 
    'Apr(?:il)?-\\d{1,2}-\\d{2}', 
    '\\d{4}\\.Oct(?:ober)?\\.\\d{1,2}', 
    '\\d{1,2}/Aug(?:ust)?/\\d{4}', 
    '\\d{2}/Jul(?:y)?/\\d{1,2}', 
    '\\d{2}/Oct(?:ober)?/\\d{1,2}', 
    '\\d{2}\\sJul(?:y)?\\s\\d{1,2}', 
    '\\d{2}/May/\\d{1,2}', 
    '\\d{1,2}\\.May\\.\\d{2}', 
    '\\d{1,2}\\sSep(?:tember)?\\s\\d{2}', 
    '\\d{2}\\.\\d{1,2}\\.\\d{1,2}', 
    '\\d{4}/Mar(?:ch)?/\\d{1,2}', 
    '\\d{4}\\sApr(?:il)?\\s\\d{1,2}', 
    '\\d{1,2}\\.Oct(?:ober)?\\.\\d{4}', 
    '\\d{1,2}\\.Aug(?:ust)?\\.\\d{2}', 
    '\\d{1,2}-Jul(?:y)?-\\d{4}', 
    '\\d{2}/Nov(?:ember)?/\\d{1,2}', 
    'Jul(?:y)?-\\d{1,2}-\\d{2}', 
    '\\d{1,2}\\.Dec(?:ember)?\\.\\d{2}', 
    'Jul(?:y)?/\\d{1,2}/\\d{2}', 
    'Mar(?:ch)?-\\d{1,2}-\\d{2}', 
    '\\d{2}\\sAug(?:ust)?\\s\\d{1,2}', 
    'Feb(?:ruary)?-\\d{1,2}-\\d{2}', 
    'Dec(?:ember)?\\.\\d{1,2}\\.\\d{2}', 
    '\\d{1,2}/Apr(?:il)?/\\d{2}', 
    '\\d{4}/Apr(?:il)?/\\d{1,2}', 
    '\\d{1,2}-Nov(?:ember)?-\\d{4}', 
    '\\d{2}-Nov(?:ember)?-\\d{1,2}', 
    '\\d{1,2}/Mar(?:ch)?/\\d{2}']

def month_to_number(month_name):
    try:
        month_number = datetime.strptime(month_name, '%b').month
    except ValueError:
        try:
            month_number = datetime.strptime(month_name, '%B').month
        except ValueError:
            month_number = 1  # 기본값
    return str(month_number).zfill(2)

# 날짜 매칭 및 표준화 함수
def match_date(text):
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            date_str = match.group()
            date_parts = re.split(r'\.|\(|\)|-|/', date_str)  # 다양한 구분자로 분리
            date_parts = [part for part in date_parts if part]  # 빈 문자열 제거

            # 연도, 월, 일 추출 및 처리
            year, month, day = '', '', ''
            for part in date_parts:
                if re.match(r'\d{4}', part):  # 연도 추출
                    year = part
                elif re.match(r'\d{1,2}', part):  # 월 또는 일 추출
                    if not month:
                        month = part.zfill(2)
                    elif not day:
                        day = part.zfill(2)
                elif re.match(r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)', part, re.IGNORECASE):  # 영어 월 추출
                    month = month_to_number(part)
            
            # 연도가 누락된 경우 현재 연도로 설정
            if not year:
                year = str(datetime.now().year)

            # 모든 부분이 제대로 추출되었는지 확인
            if year and month and day:
                return f"{year}.{month}.{day}"

    return None


def process_text(text):
    if '~' in text:
        start_text, end_text = text.split('~', 1)
        start_date_str = match_date(start_text)
        end_date_str = match_date(end_text)
        if start_date_str and end_date_str:
            return start_date_str + '~' + end_date_str
        elif start_date_str:
            return start_date_str
        elif end_date_str:
            return end_date_str
    else:
        return match_date(text)
            

for result in results:
    for detection in result:
        box = detection['box']
        text = detection['text']
        confidence = detection['confidence']
        
        if match_date(text):
            print(f"일정 발견 : {text}\n   => {process_text(text)}\n")

# texts = ["24.01.12(2)18:00","2024.01.17()","2024.01.19(3)","24.01.12(2) 18:00","2024.01.17()","2024.01.19(3)","24.01.12(2)18:00"
#         ,"24.01.12(금) 18.00","2024.01.17(수)","2024.01.19금)","24,01,12(금) 18.00","2024.01.17(수)"]
# for text in texts:
#     print(f"일정 발견 : {text}\n   => {process_text(text)}\n")

runtime = time.time() - start
print("Runtime : " + str(runtime) + "s")