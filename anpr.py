import cv2
import matplotlib.pyplot as plt
import pytesseract
import sys

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
char_whitelist = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
plot = 1

def print_ascii(text):
    ascii_text = ""
    for char in text:
        ascii_text += str(ord(char)) + " "
    print("ASCII representation:", ascii_text)

def format_text(text):
    replaced_text = ""
    for char in text:
        # Check if the ASCII value of the character is not 12
        if ord(char) != 12 and ord(char) != 10 and ord(char) != 32:
            replaced_text += char
    return replaced_text

def plot_images(img1, img2, title1="", title2=""):
    if plot == 0:
        return
    fig = plt.figure(figsize=[15,15])
    ax1 = fig.add_subplot(121)
    ax1.imshow(img1, cmap="gray")
    ax1.set(xticks=[], yticks=[], title=title1)

    ax2 = fig.add_subplot(122)
    ax2.imshow(img2, cmap="gray")
    ax2.set(xticks=[], yticks=[], title=title2)
    plt.show()

def extraction(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plot_images(image, gray)
    blur = cv2.bilateralFilter(gray, 11,90, 90)
    plot_images(gray, blur)
    edges = cv2.Canny(blur, 30, 200)
    plot_images(blur, edges)
    cnts, new = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = image.copy()

    _ = cv2.drawContours(image_copy, cnts, -1, (255,0,255),2)
    plot_images(image, image_copy)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    image_copy = image.copy()
    _ = cv2.drawContours(image_copy, cnts, -1, (255,0,255),2)
    plot_images(image, image_copy)

    plate = None
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        edges_count = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        x,y,w,h = cv2.boundingRect(c)
        if len(edges_count) == 4:
            x,y,w,h = cv2.boundingRect(c)
            plate = gray[y:y+h, x:x+w]
            cv2.imshow('frame',plate)
            cv2.waitKey(0)
            text = pytesseract.image_to_string(plate, config=char_whitelist)
            text = format_text(text)
            if text != "":
            	print(text)
            	print_ascii(text)
            	break
    return plate

imagePath = "photos/" + sys.argv[1]
image = cv2.imread(imagePath)
plate = extraction(image)

cv2.destroyAllWindows()
