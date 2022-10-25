# Import required packages
import cv2
import pytesseract
import numpy as np
from PIL import Image
from datetime import datetime, timedelta 
import os
import json
# import easyocr

def exists(digit_res, thresh=0.8):
	loc = np.where(digit_res >= thresh)

	if len(loc[-1]) == 0:
		return False

	for pt in zip(*loc[::-1]):
		if digit_res[pt[1]][pt[0]] == 1:
			return False

	return True

def show_image(img):
	cv2.imshow("Image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# Mention the installed location of Tesseract-OCR in your system
tesseract_exe_path = 'D:\\Programs\\Tesseract OCR\\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = tesseract_exe_path

csv = "recognized.csv"
if not os.path.isfile(csv):
	# A text file is created and flushed
	file = open(csv, "w+")
	file.write("log date, location(true|false), capture date, within 5 minutes(true|false)\n")
	file.close()

# Read image from which text needs to be extracted
raw = cv2.imread('hk-id-card-sample.jpg')
# Convert the image to gray scale
gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

# Performing OTSU threshold
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# # Specify structure shape and kernel size.
# # Kernel size increases or decreases the area
# # of the rectangle to be detected.
# # A smaller value like (10, 10) will detect
# # each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

# Finding contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
# print(contours)
# Creating a copy of image
# im2 = crop_img.copy()
im2 = gray.copy()

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
	cropped = im2[y:y + h, x:x + w]
	lang = "eng"
	if "lang" in config:
		lang = config["lang"]
	print("lang",lang)
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
		print("img_2_string_config",img_2_string_config)
	text_denoised = pytesseract.image_to_string(cropped, lang = lang, config=img_2_string_config)
	text_denoised = text_denoised.replace("\n","").strip()
	print("text_denoised,", text_denoised)

	# easy_denoised = reader.readtext(denoised, detail = 0)
	# print("easy_denoised,", easy_denoised)


	# se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
	# bg=cv2.morphologyEx(denoised, cv2.MORPH_DILATE, se)
	# out_gray=cv2.divide(denoised, bg, scale=255)
	# cv2.imwrite(key+"_out_gray.png", out_gray)
	# text_out_gray = pytesseract.image_to_string(out_gray, lang = lang)
	# text_out_gray = text_out_gray.replace("\n","").strip()
	# print("text_out_gray", text_out_gray)

	# easy_out_gray = reader.readtext(out_gray, detail = 0)
	# print("easy_out_gray,", easy_out_gray)

	rect = cv2.rectangle(raw, (x, y), (x + w, y + h), (0, 255, 0), 2)
	value = text_denoised
	# if text_denoised == "":
	# 	value = text_out_gray
	
	dictionary[key] = value
	# print("dictionary[key]", dictionary[key])

with open("output.json", "w") as outfile:
	json.dump(dictionary, outfile)

# base_minute_buffer = 5

# Looping through the identified contours
# Then rectangular part is cropped and passed on
# to pytesseract for extracting text from it
# Extracted text is then written into the text file
# for cnt in contours:
# 	x, y, w, h = cv2.boundingRect(cnt)
# 	# Cropping the text block for giving input to OCR
# 	cropped = im2[y:y + h, x:x + w]
# 	# Apply OCR on the cropped image
# 	# , lang = "chi_tra"
# 	text = pytesseract.image_to_string(cropped)
# 	text = text.replace("\n","").strip()

# 	if text != "" :
# 		print("text", text)
# 		# Drawing a rectangle on copied image
# 		rect = cv2.rectangle(raw, (x, y), (x + w, y + h), (0, 0, 255), 2)
# 		print(x, y, w, h)


show_image(raw)