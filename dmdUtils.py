import numpy as np
import time
import serial
import tifffile
from PIL import Image
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import rotate
from skimage.draw import disk
from skimage import transform as tf
from skimage import draw, io
from scipy.ndimage import affine_transform

def initialize_led_connection():
    s1 = serial.Serial(port = 'COM3', baudrate = 57600)
    time.sleep(1)
    return(s1)

def v_650(s1, volt):
    s1.write(str.encode('DAC2, %s\r\n'%volt))

def v_750(s1, volt):
    s1.write(str.encode('DAC1, %s\r\n'%volt))

def np_to_dmd(nparray):
    '''
    takes a binary numpy array and saves it as the source for the DMD bufData
    '''
    nparray8 = nparray.astype(np.bool)
    bin_array = np.packbits(nparray8)
    bin_path = r'C:\\DMD_CTRL_NO_TOUCH\\Functional Mosaic Control\\bufData.bin'
    bin_file = open(bin_path, 'w')
    bin_array.tofile(bin_file, sep='')

def dmd_to_np():
    '''
    returns current pattern in the DMD bin file as a numpy array
    '''
    bin_path = r'C:\\DMD_CTRL_NO_TOUCH\\Functional Mosaic Control\\bufData.bin'

    try:
        dmd_array = np.unpackbits(np.fromfile(bin_path, dtype=np.uint8)).reshape(600,800)
    except:
        dmd_array = np.unpackbits(np.fromfile(bin_path, dtype=np.uint8)).reshape(128,600,800)

    return(dmd_array.astype(np.bool))

def convert_to_frequency(image):
    '''
    takes in a numpy array of floats between 1 and 0 and converts them into dithering frequencies
    '''
    result = np.zeros((128,600,800), dtype = np.uint8)
    for i in range(128):
        result[i] = (image > i/128.)
    randomized = np.random.permutation(result)

    return(randomized)

def mickey_mouse():#s1, v):
    #v_650(s1, v)
    #v_750(s1, 0)

    print('Focus the cartoon mouse by adjusting the internal DMD tube lens')

    mick_path = r'C:\Data\Jason\SCRIPTS\mickeymouse.png'
    img = Image.open(mick_path).convert('LA')
    np_img = np.array(img)[:,:,0]

    bin_img = np.array(np_img < 50, dtype = int)

    zoomed_img = zoom(bin_img, 2, order = 0)

    h, w = zoomed_img.shape
    border_w = int(800-w)//2
    border_h = int(600-h)//2
    pattern = np.zeros((600,800))
    pattern[border_h:600-border_h, border_w:800-border_w] = zoomed_img

    pattern = np.array(pattern, dtype = np.bool)
    np_to_dmd(pattern)

def blank_pattern(s1, v):
    v_650(s1, v)
    v_750(s1, 0)

    pattern = np.zeros((600,800))

    pattern = np.array(pattern, dtype = np.uint8)
    np_to_dmd(pattern)

def blank_pattern_all_on(s1, v):
    v_650(s1, v)
    v_750(s1, 0)

    pattern = np.ones((600,800))

    pattern = np.array(pattern, dtype = np.uint8)
    np_to_dmd(pattern)

def rotateImage(img, angle, pivot):
    '''
    rotates an image by angle about a pivot point(x,y)
    '''
    padX = [img.shape[1]-pivot[0], pivot[0]]
    padY = [img.shape[0]-pivot[1], pivot[1]]
    imgP = np.pad(img, [padY,padX], 'constant')
    imgR = rotate(imgP, angle, reshape = False)
    return(imgR[padY[0] : -padY[1], padX[0] : -padX[1]])

def calibration_protocol(s1):
    print('go to 488 LP mode in micromanager')
    input('press enter to continue')

    v_650(s1, 20000)
    v_750(s1, 20000)

    pattern_1 = np.zeros((600,800), dtype = np.uint8)
    pattern_1[140:160,140:160] = 1
    np_to_dmd(pattern_1)
    print('Snap an image\n')
    x1 = int(input('x value of spot center:\n'))
    y1 = int(input('y_value of spot center:\n'))

    pattern_2 = np.zeros((600,800), dtype = np.uint8)
    pattern_2[440:460,440:460] = 1
    np_to_dmd(pattern_2)
    print('Snap another image\n')
    x2 = int(input('x value of spot center:\n'))
    y2 = int(input('y_value of spot center:\n'))

    pattern_3 = np.zeros((600,800), dtype = np.uint8)
    pattern_3[140:160,440:460] = 1
    np_to_dmd(pattern_3)
    print('Snap another image\n')
    x3 = int(input('x value of spot center:\n'))
    y3 = int(input('y_value of spot center:\n'))

    pattern_4 = np.zeros((600,800), dtype = np.uint8)
    pattern_4[440:460,140:160] = 1
    np_to_dmd(pattern_4)
    print('Snap another image\n')
    x4 = int(input('x value of spot center:\n'))
    y4 = int(input('y_value of spot center:\n'))

    recorded_values = open('C:\\Data\\Jason\\SCRIPTS\\calibration_coordinates.txt', 'w')
    for val in [x1, y1, x2, y2, x3, y3, x4, y4]:
        recorded_values.write(str(val) + '\n')

def img_from_dmd(x, y):
    x1, y1, x2, y2, x3, y3, x4, y4 = 150, 150, 450, 450, 450, 150, 150, 450
    src = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape((4,2))

    recorded_values = open('C:\\Data\\Jason\\SCRIPTS\\calibration_coordinates.txt', 'r')
    x1, y1, x2, y2, x3, y3, x4, y4 = [int(i.strip('\n')) for i in recorded_values.readlines()]
    dst = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape((4,2))

    tform = tf.estimate_transform('affine', src, dst)

    x_in_img, y_in_img = tform([x, y])[0] # 210502 had to switch y and x - possibly due to update in skimage?

    return(x_in_img, y_in_img)

def dmd_from_img(x, y):
    x1, y1, x2, y2, x3, y3, x4, y4 = 150, 150, 450, 450, 450, 150, 150, 450
    src = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape((4,2))

    recorded_values = open('C:\\Data\\Jason\\SCRIPTS\\calibration_coordinates.txt', 'r')
    x1, y1, x2, y2, x3, y3, x4, y4 = [int(i.strip('\n')) for i in recorded_values.readlines()]
    dst = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape((4,2))

    tform = tf.estimate_transform('affine', src, dst)

    x_in_dmd, y_in_dmd = tform.inverse([x, y])[0]

    return(x_in_dmd, y_in_dmd)


def project_circle(s1, x_in_img, y_in_img, size, intensity):

    pattern = np.zeros((600,800))

    x_in_dmd, y_in_dmd = dmd_from_img(x_in_img, y_in_img)

    rr, cc = disk((np.round(y_in_dmd), np.round(x_in_dmd)), size, shape = pattern.shape)
    pattern[rr, cc] = intensity

    pattern = convert_to_frequency(pattern)

    np_to_dmd(pattern)

def add_circle(s1, x_in_img, y_in_img, size, intensity):

    pattern = dmd_to_np().astype(float)

    if len(pattern.shape) == 3:
        pattern = np.average(pattern, axis = 0)

    x_in_dmd, y_in_dmd = dmd_from_img(x_in_img, y_in_img)

    rr, cc = disk((np.round(y_in_dmd), np.round(x_in_dmd)), size, shape = (600,800))
    pattern[rr, cc] = intensity

    pattern = convert_to_frequency(pattern)

    np_to_dmd(pattern)

def project_circle_inv(s1, x_in_img, y_in_img, size, intensity):

    pattern = np.zeros((600,800))

    x_in_dmd, y_in_dmd = dmd_from_img(x_in_img, y_in_img)
    print(x_in_dmd, y_in_dmd)

    rr, cc = disk((np.round(y_in_dmd), np.round(x_in_dmd)), size, shape = pattern.shape)
    pattern[rr, cc] = intensity

    pattern = convert_to_frequency(1-pattern)
    #pattern = np.array(pattern, dtype = np.uint8)

    np_to_dmd(pattern)


def project_square(s1, x_in_img, y_in_img, size):

    pattern = np.zeros((600,800))

    x_in_dmd, y_in_dmd = dmd_from_img(x_in_img, y_in_img)

    pattern[y_in_dmd - size//2: y_in_dmd + size//2, x_in_dmd - size//2: x_in_dmd + size//2] = 1

    pattern = np.array(pattern, dtype = np.uint8)


def predict_DMD_image():

    if len((dmd_to_np().shape)) == 3:
        dmd_array = np.average(dmd_to_np(), axis = 0) > 0
    else:
        dmd_array = dmd_to_np()

    x1, y1, x2, y2, x3, y3, x4, y4 = 150, 150, 450, 450, 450, 150, 150, 450
    src = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape((4,2))

    recorded_values = open('C:\\Data\\Jason\\SCRIPTS\\calibration_coordinates.txt', 'r')
    x1, y1, x2, y2, x3, y3, x4, y4 = [int(i.strip('\n')) for i in recorded_values.readlines()]
    dst = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape((4,2))

    tform_inv = tf.estimate_transform('affine', np.roll(dst, 1, axis = 1), np.roll(src, 1, axis = 1)) # roll to put ys first for image transformations
    inv_affine_matrix = tform_inv.params

    predicted_dmd_image = affine_transform(dmd_array.astype(float), inv_affine_matrix, order = 0, output_shape = (1024,1024)).astype(np.int16)

    #tifffile.imwrite(save_path, predicted_dmd_image, metadata = description)
    
    return(predicted_dmd_image)