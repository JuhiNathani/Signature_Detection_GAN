# This generates new signature images 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil


def add_random_straight_lines(image,height, width):
    ''' Add random straight lines to the image '''
    num_lines = np.random.randint(1, 5) # number of lines to be added
    y0 = int(height/num_lines) # gap/width between each lines
    for i in range(num_lines):
        line_thickness = np.random.randint(1, 5)
        x1, x2 = 0, width # starting and ending x coordinates
        y = y0*(i+1) + np.random.randint(-0.05*height, 0.05*height) # y coordinate of line
        image = cv2.line(image, (x1, y), (x2, y), (0, 0, 0), thickness=line_thickness) #draw line
        prev_y = y
    return image

def add_random_text(image, height, width):
    ''' Add random texts to the image '''
    closings = ['Sincerly', 'Regards', 'Yours truly', 'Best regards', 'Cordially']
    bottom_text = ['Amal Joseph', 'Steve Jobs', 'Larry Page', 'Paul Walker', 'Raja Ravi Varma', 'Katie Bouman', 'Ada Loveless']

    font = [cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL]
    y = np.random.randint(0.75*height, 1.02*height)
    x = np.random.randint(0.0005*width, 0.3*width)
    fontScale = np.random.random() + 0.7
    thickness = np.random.randint(1, 3)
    image = cv2.putText(image, np.random.choice(bottom_text), (x, y), np.random.choice(font), fontScale, (0, 0, 0), thickness, cv2.LINE_AA)
    return image

def process_image(image_path):
    ''' Add random straight lines and texts to the image '''
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    image = add_random_text(image, height, width)
    image = add_random_straight_lines(image, height, width)
    return image

# move images from sub directories to another folder
root_path = './images/'

for root, dirs, files in os.walk(root_path):
    for filename in files:
        shutil.copy(os.path.join(root, filename), './gan-sign_data_kaggle/A/')

# apply image augumentations and save them
root_path = './gan-sign_data_kaggle/A/'
print(os.walk(root_path))

for root, dirs, files in os.walk(root_path):
    for filename in files:
        image = process_image(os.path.join(root, filename))
        plt.imshow(image)
        cv2.imwrite(f'./gan-sign_data_kaggle/B/{filename}', image)

      

# splitting into train and test
root = './gan-sign_data_kaggle/'
srcA_path = './gan-sign_data_kaggle/A/'
srcB_path = './gan-sign_data_kaggle/B/'

trainA_path = './gan_signdata_kaggle_TT_Split/trainA/'
testA_path = './gan_signdata_kaggle_TT_Split/testA/'


trainB_path = './gan_signdata_kaggle_TT_Split/trainB/'
testB_path = './gan_signdata_kaggle_TT_Split/testB/'

def split_data(src_path, train_path, test_path, split_ratio):
    files = np.array(os.listdir(src_path))
    np.random.shuffle(files)
    split_index = int(split_ratio * len(files))
    testA = files[0:split_index]
    trainA = files[split_index:]
    [shutil.copy(os.path.join(src_path, path), os.path.join(train_path, path)) for path in trainA]
    [shutil.copy(os.path.join(src_path, path), os.path.join(test_path, path)) for path in testA]

split_data(srcA_path, trainA_path, testA_path, 0.1)
split_data(srcB_path, trainB_path, testB_path, 0.1)

#os.rmdir(srcA_path)
#os.rmdir(srcB_path)
#os.rmdir('./gan-sign_data_kaggle/images/')

##################### TRANSFORMING DATA SO THAT IT CAN BE INPUT TO GAN ####################### 

import os
from PIL import Image
im_size = 512
def make_square(image, min_size=512, fill_color=(255, 255, 255, 0)):
    ''' Resize image as a square with signature in the center and black(transparent) strips at top and bottom. '''
    x, y = image.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(image, (int((size - x) / 2), int((size - y) / 2)))
    new_im = new_im.resize((im_size, im_size))
    return new_im

def resize_images(path):
    ''' Function to resize the images to the ip format for gans. '''
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(path+item):
            image = Image.open(path+item)
            image = make_square(image)
            image.save(path+item)

resize_images(trainA_path)
resize_images(trainB_path)
resize_images(testA_path)
resize_images(testB_path)