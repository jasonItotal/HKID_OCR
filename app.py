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
import re
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms  # , utils

from u2net_data_loader import RescaleT
from u2net_data_loader import ToTensorLab
from u2net_data_loader import SalObjDataset
from PIL import Image
import glob

from model import U2NETP  # small version u2net 4.7 MB

from flask import Flask, request, Response, jsonify

# need to run only once to download and load model into memory
ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)

# Mention the installed location of Tesseract-OCR in your system
tesseract_exe_path = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = tesseract_exe_path


def image_to_box(img):
    boxes = pytesseract.image_to_boxes(img, lang='eng')  # Run tesseract, returning the bounding boxes

    h, w, _ = img.shape # assumes color image
    mask = np.zeros((h, w), np.uint8)

    # Fill the bounding boxes on the image
    for b in boxes.splitlines():
        b = b.split(' ')
        mask = cv2.rectangle(mask, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), 255, -1)

    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))  # Dilate the boxes in the mask
    
    show_image(mask)

def check_template(image, template="old_card_template.png"):
    print("in check template")
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template, 0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    pt_list = zip(*loc[::-1])

    result = True
    # print(F"len(list(pt_list)):{len(list(pt_list))}")
    if len(list(pt_list)) == 0:
        result = False
    # for pt in pt_list:
    #     cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    # show_image(image)
    # print(result)
    return result

def is_old_card(source):
    image = source.copy()
    is_old_card = check_template(image)
    rotated = False
    if not is_old_card:
        is_old_card = check_template(
            image, template="old_card_template_rotated.png")
        if is_old_card:
            rotated = True
        # image = cv2.rotate(image, cv2.ROTATE_180)
        # show_image(image)
        
    print('is_old_card')
    print(is_old_card)
    return [is_old_card,rotated]

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def u2net_remove_background(source_path):
    # --------- 1. get image path and name ---------
    model_name='u2netp'# fixed as u2netp
    # changed to 'images' directory which is populated while running the script
    # image_dir = os.path.join(os.getcwd(), 'images')


    # path to u2netp pretrained weights
    model_dir = os.path.join(os.getcwd(), model_name + '.pth')

    img_name_list = glob.glob(os.getcwd() + os.sep + source_path)
    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    net = U2NETP(3, 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))

    net.eval()

    for i_test, data_test in enumerate(test_salobj_dataloader):
        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)
        source = cv2.imread(source_path)
        result = u2net_result(source, pred)
        del d1,d2,d3,d4,d5,d6,d7
    return result

def u2net_result(image, pred):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    resized = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    return cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR)

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


# input source_image output a wrapped image
def wrap_image(process_image_path):
    source_image = cv2.imread(process_image_path)
    # height, width, _ = source_image.shape

    mask = u2net_remove_background(process_image_path)
    source_image = cv2.bitwise_and(source_image, mask)
    # rotate image
    image = source_image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

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
    if displayCnt is not None:
        # Obtain birds' eye view of image
        return four_point_transform(image, displayCnt.reshape(4, 2))
    else:
        return image
 # identify process


def identify_hkid(source_image):
    source_image = cv2.resize(source_image, dsize=(476, 300),
                        interpolation=cv2.INTER_NEAREST)

    im2 = source_image.copy()

    config_json = 'new_format.json'
    result = is_old_card(im2)
    # result[0] check is old card
    if result[0]:
        print("old card detected.")
        config_json = 'old_format.json'

    # result[1] check is rotated
    if result[1]:
        im2 = cv2.rotate(im2,cv2.ROTATE_180)
        
    
    gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    jsonFile = open(config_json)
    locations = json.loads(jsonFile.read())

    # reader = easyocr.Reader(['ch_tra','en'])

    # move to output
    # print("os.getcwd()",os.getcwd())
    os.chdir(".\\output")
    dictionary = {}
    # print(locations)
    for key in locations:
        config = locations[key]
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
            
        if key == 'name' or key == 'name_en':
            cropped = cv2.blur(cropped, (2, 2))
        cv2.imwrite(key+".png", cropped)
        img_2_string_config = ''
        if "img_2_string_config" in config:
            img_2_string_config = config["img_2_string_config"]
            # text_denoised = pytesseract.image_to_string(cropped, lang = lang, config=img_2_string_config)
        # print("lang", lang)
        if lang == "eng":
            text = pytesseract.image_to_string(
                cropped, lang=lang, config=img_2_string_config)
        else:
            # print("ocr")
            result = ocr.ocr(cropped,  det=False)
            text = result[0][0][0]
            # print("text before sub", text)
            text = re.sub(r'[a-z0-9A-Z]+', '', text, re.I)

        text_denoised = text.replace("\n", "").strip()
        print(F"ocr {key:>24} :", text_denoised)

        rect = cv2.rectangle(im2, (x, y),
                             (x + w, y + h), (0, 255, 0), 2)
        value = text_denoised
        # if text_denoised == "":
        # 	value = text_out_gray

        dictionary[key] = value
        # print("key,", value)
        # print("dictionary[key]", dictionary[key])

    ##write result as json file
    with open("output.json", "w", encoding="utf-8") as outfile:
        json.dump(dictionary, outfile, ensure_ascii=False)

    # print("dictionary", dictionary)

    cv2.imwrite("detect_zone.png", im2)
    cv2.imwrite("source.png", source_image)
    
    os.chdir("..")
    # show_image(im2)
    return dictionary
    

if __name__ == '__main__':
    process_image_path = "received\\download.jfif"
    # image.save(process_image_path)
    do_wrapped = True
    if do_wrapped:
        process_image = wrap_image(process_image_path)
    else:
        process_image = cv2.imread(process_image_path)
    
    hkid_info = identify_hkid(process_image)
    process_time = time.process_time()
    t_sec = round(process_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60) 
    
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))
    print("process_image_path")
    file_path = os.path.join(os.getcwd(),process_image_path)
    print(file_path)
    if os.path.exists(file_path):
        print("process file exist")
        print("clear process file")
        os.remove(file_path)
    else:
        print("process file not exist")

app = Flask(__name__)

@app.route('/hkid_ocr', methods=['POST'])
def hkid_ocr():
    image = request.files['hkid']
    process_image_path = "received\\hkid.jpg"
    image.save(process_image_path)
    do_wrapped = True
    if do_wrapped:
        process_image = wrap_image(process_image_path)
    else:
        process_image = cv2.imread(process_image_path)
    
    hkid_info = identify_hkid(process_image)
    process_time = time.process_time()
    t_sec = round(process_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60) 
    
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))
    print("process_image_path")
    file_path = os.path.join(os.getcwd(),process_image_path)
    print(file_path)
    if os.path.exists(file_path):
        print("process file exist")
        print("clear process file")
        os.remove(file_path)
    else:
        print("process file not exist")
    return jsonify(
        hkid_info
    )