import cv2 as cv
import os
import re

def fname_to_int(fname):
    return int(re.sub('\D', '', fname))

def frames_to_vid(image_folder, output_name):
    images = [img for img in os.listdir(image_folder) if img.endswith('.png')]
    images.sort(key=fname_to_int)

    frame = cv.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape 

    FOURCC = cv.VideoWriter_fourcc(*'XVID')
    video = cv.VideoWriter(output_name, FOURCC, 30, (width, height))

    for img in images:
        video.write(cv.imread(os.path.join(image_folder, img)))
    
    video.release()
