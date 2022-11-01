# Import required packages
import cv2
import pytesseract
import numpy as np
import subprocess
# from cnocr import CnOcr
# import easyocr
from paddleocr import PaddleOCR
import os
import time
import json
from imutils.perspective import four_point_transform


def exists(digit_res, thresh=0.8):
    loc = np.where(digit_res >= thresh)

    if len(loc[-1]) == 0:
        return False

    for pt in zip(*loc[::-1]):
        if digit_res[pt[1]][pt[0]] == 1:
            return False

    return True


def remove_zero_pad(image):
    dummy = np.argwhere(image != 0)  # assume blackground is zero
    max_y = dummy[:, 0].max()
    min_y = dummy[:, 0].min()
    min_x = dummy[:, 1].min()
    max_x = dummy[:, 1].max()
    crop_image = image[min_y:max_y, min_x:max_x]

    return crop_image


def show_image(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cmd(command, timeout=5):
    subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, encoding="utf-8")
    subp.wait(timeout)
    if subp.poll() == 0:
        print(subp.communicate()[1])
    else:
        print("failed")


# need to run only once to download and load model into memory
ocr = PaddleOCR(use_angle_cls=True, lang="ch")

# Mention the installed location of Tesseract-OCR in your system
tesseract_exe_path = 'D:\\Programs\\Tesseract OCR\\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = tesseract_exe_path

# csv = "recognized.csv"
# if not os.path.isfile(csv):
# 	# A text file is created and flushed
# 	file = open(csv, "w+")
# 	file.write("log date, location(true|false), capture date, within 5 minutes(true|false)\n")
# 	file.close()

# Read image from which text needs to be extracted
raw = cv2.imread('hkid_realtest2.jpg')
height, width, channels = raw.shape

resize_height = height
resize_width = width
# pic is long pic
if height > width and height > 1000:
    resize_height = int(1000)
    resize_width = int(width*resize_height/height)
elif width > height and width > 1000:
    resize_width = int(1000)
    resize_height = int(height*resize_width/width)

print(height, width)
print(resize_height, resize_width)

raw = cv2.resize(raw, dsize=(resize_width, resize_height),
                 interpolation=cv2.INTER_NEAREST)

gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

os.chdir(".\\U-2-Net")
ts = time.time()
tsString = str(ts).replace(".", "_")
tsString = "1666756920_1268156"
print(tsString)
processPath = os.path.join(os.getcwd(), 'images', tsString+".png")
cv2.imwrite(processPath, raw)
# print("os.getcwd()",os.getcwd())
cmd("python u2net_test.py", 30)
maskPath = os.path.join(os.getcwd(), 'results', tsString+".png")
if os.path.exists(maskPath):
    # print("result exist")
    # clear process file
    # if os.path.exists(processPath):
    #     os.remove(processPath)

    mask = cv2.imread(maskPath)
    raw = cv2.bitwise_and(raw, mask)
    # show_image(masked)
    os.chdir("..")
else:
    print("error: process image failed in mask.")
    exit()

# rotate image
image = raw
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Find contours and sort for largest contour
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None

for c in cnts:
    # Perform contour approximation
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        displayCnt = approx
        break

# Obtain birds' eye view of image
warped = four_point_transform(image, displayCnt.reshape(4, 2))

# cv2.imshow("thresh", thresh)
# cv2.imshow("warped", warped)
# cv2.waitKey()

# wrap image
# img = raw

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # blur image
# blur = cv2.GaussianBlur(gray, (3, 3), 0)

# # do otsu threshold on gray image
# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

# # apply morphology
# kernel = np.ones((7, 7), np.uint8)
# morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
# morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

# # get largest contour
# contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours = contours[0] if len(contours) == 2 else contours[1]
# area_thresh = 0
# for c in contours:
#     area = cv2.contourArea(c)
#     if area > area_thresh:
#         area_thresh = area
#         big_contour = c

# # draw white filled largest contour on black just as a check to see it got the correct region
# page = np.zeros_like(img)
# cv2.drawContours(page, [big_contour], 0, (255, 255, 255), -1)

# # get perimeter and approximate a polygon
# peri = cv2.arcLength(big_contour, True)
# corners = cv2.approxPolyDP(big_contour, 0.04 * peri, True)

# # draw polygon on input image from detected corners
# polygon = img.copy()
# cv2.polylines(polygon, [corners], True, (0, 0, 255), 1, cv2.LINE_AA)
# # Alternate: cv2.drawContours(page,[corners],0,(0,0,255),1)

# # print the number of found corners and the corner coordinates
# # They seem to be listed counter-clockwise from the top most corner
# # print(len(corners))
# # print(corners)

# # for simplicity get average of top/bottom side widths and average of left/right side heights
# # note: probably better to get average of horizontal lengths and of vertical lengths
# width = 0.5*((corners[0][0][0] - corners[1][0][0]) +
#              (corners[3][0][0] - corners[2][0][0]))
# height = 0.5*((corners[2][0][1] - corners[1][0][1]) +
#               (corners[3][0][1] - corners[0][0][1]))
# width = np.int0(width)
# height = np.int0(height)

# # reformat input corners to x,y list
# icorners = []
# for corner in corners:
#     pt = [corner[0][0], corner[0][1]]
#     icorners.append(pt)
# icorners = np.float32(icorners)

# # get corresponding output corners from width and height
# ocorners = [[width, 0], [0, 0], [0, height], [width, height]]
# ocorners = np.float32(ocorners)

# # get perspective tranformation matrix
# M = cv2.getPerspectiveTransform(icorners, ocorners)

# # do perspective
# warped = cv2.warpPerspective(img, M, (width, height))

# show_image(warped)


# identify process
warped = cv2.resize(warped, dsize=(476, 300), interpolation=cv2.INTER_NEAREST)
source = warped.copy()

im2 = source.copy()

gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

jsonFile = open('new_format.json')
locations = json.loads(jsonFile.read())

# reader = easyocr.Reader(['ch_tra','en'])

# move to output
# print("os.getcwd()",os.getcwd())
os.chdir(".\\output")
dictionary = {}
print(locations)
for key in locations:
    config = locations[key]
    # print("key:")
    print(key)
    rect_buffer = 5
    # print("config:")
    x = config["x"] - rect_buffer
    y = config["y"] - rect_buffer
    w = config["w"] + rect_buffer
    h = config["h"] + rect_buffer
    cropped = gray[y:y + h, x:x + w]
    lang = "eng"
    if "lang" in config:
        lang = config["lang"]
    # cropped2x = waifu2x.process(cropped)
    # show_image(cropped)
    # cv2.imwrite(key+".png", cropped)
    # text = pytesseract.image_to_string(cropped, lang = lang)
    # text = text.replace("\n","").strip()
    # cropped = cv2.resize(cropped, dsize=(w * 2, h * 2), interpolation=cv2.INTER_NEAREST)

    cropped = cv2.fastNlMeansDenoising(cropped)
    cv2.imwrite(key+"_denoised.png", cropped)
    img_2_string_config = ''
    if "img_2_string_config" in config:
        img_2_string_config = config["img_2_string_config"]
        # text_denoised = pytesseract.image_to_string(cropped, lang = lang, config=img_2_string_config)
    print("lang", lang)
    if lang == "eng":
        print("pytesseract")
        print("img_2_string_config", img_2_string_config)
        text = pytesseract.image_to_string(
            cropped, lang=lang, config=img_2_string_config)
    else:
        print("ocr")
        result = ocr.ocr(cropped,  det=False)
        text = result[0][0][0]

    text_denoised = text.replace("\n", "").strip()
    print("text_denoised,", text_denoised)

    rect = cv2.rectangle(source, (x, y), (x + w, y + h), (0, 255, 0), 2)
    value = text_denoised
    # if text_denoised == "":
    # 	value = text_out_gray
    
    dictionary[key] = value
    print("key,", value)
    # print("dictionary[key]", dictionary[key])

with open("output.json", "w") as outfile:
    json.dump(dictionary, outfile)

print("dictionary", dictionary)

cv2.imwrite("source.png", source)
cv2.imwrite("warped.png", warped)
show_image(source)
