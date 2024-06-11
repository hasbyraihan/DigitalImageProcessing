import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from collections import Counter
from pylab import savefig
import cv2
import os
import json
from dotenv import dotenv_values
from dotenv import load_dotenv
import Levenshtein
from sklearn.cluster import DBSCAN
load_dotenv()

# Load environment variables from .env file
config = dotenv_values(".env")

def grayscale():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r = img_arr[:, :, 0]
    g = img_arr[:, :, 1]
    b = img_arr[:, :, 2]
    new_arr = r.astype(int) + g.astype(int) + b.astype(int)
    new_arr = (new_arr/3).astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def is_grey_scale(img_path):
    im = Image.open(img_path).convert('RGB')
    w, h = im.size
    for i in range(w):
        for j in range(h):
            r, g, b = im.getpixel((i, j))
            if r != g != b:
                return False
    return True

def binarize(threshold=128):
    img = Image.open("static/img/img_now.jpg").convert('L')  # Convert to grayscale
    img_arr = np.asarray(img)
    binary_arr = np.where(img_arr >= threshold, 255, 0)  # Apply threshold
    new_img = Image.fromarray(binary_arr.astype('uint8'))
    new_img.save("static/img/img_now_binary.jpg")

def dilate():
    img = Image.open("static/img/img_now.jpg").convert('L')  # Convert to grayscale
    img_arr = np.array(img)
    h, w = img_arr.shape

    # Define the structuring element for dilation (3x3 square)
    selem = np.ones((3, 3), dtype=np.uint8)

    # Create an empty array to store the dilated image
    dilated_arr = np.zeros((h, w), dtype=np.uint8)

    # Perform dilation
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # Dilate if any pixel in the neighborhood is non-zero
            if np.any(img_arr[i - 1:i + 2, j - 1:j + 2] != 0):
                dilated_arr[i, j] = 255

    new_img = Image.fromarray(dilated_arr)
    new_img.save("static/img/img_now.jpg")


def erode():
    img = Image.open("static/img/img_now.jpg").convert('L')  # Convert to grayscale
    img_arr = np.array(img)
    h, w = img_arr.shape

    # Define the structuring element for erosion (3x3 square)
    selem = np.ones((3, 3), dtype=np.uint8)

    # Create an empty array to store the eroded image
    eroded_arr = np.zeros((h, w), dtype=np.uint8)

    # Perform erosion
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # Erode if all pixels in the neighborhood are non-zero
            if np.all(img_arr[i - 1:i + 2, j - 1:j + 2] != 0):
                eroded_arr[i, j] = 255

    new_img = Image.fromarray(eroded_arr)
    new_img.save("static/img/img_now.jpg")


def zoomin():
    img = Image.open("static/img/img_now.jpg")
    img = img.convert("RGB")
    img_arr = np.asarray(img)
    new_size = ((img_arr.shape[0] * 2),
                (img_arr.shape[1] * 2), img_arr.shape[2])
    new_arr = np.full(new_size, 255)
    new_arr.setflags(write=1)

    r = img_arr[:, :, 0]
    g = img_arr[:, :, 1]
    b = img_arr[:, :, 2]

    new_r = []
    new_g = []
    new_b = []

    for row in range(len(r)):
        temp_r = []
        temp_g = []
        temp_b = []
        for i in r[row]:
            temp_r.extend([i, i])
        for j in g[row]:
            temp_g.extend([j, j])
        for k in b[row]:
            temp_b.extend([k, k])
        for _ in (0, 1):
            new_r.append(temp_r)
            new_g.append(temp_g)
            new_b.append(temp_b)

    for i in range(len(new_arr)):
        for j in range(len(new_arr[i])):
            new_arr[i, j, 0] = new_r[i][j]
            new_arr[i, j, 1] = new_g[i][j]
            new_arr[i, j, 2] = new_b[i][j]

    new_arr = new_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def zoomout():
    img = Image.open("static/img/img_now.jpg")
    img = img.convert("RGB")
    x, y = img.size
    new_arr = Image.new("RGB", (int(x / 2), int(y / 2)))
    r = [0, 0, 0, 0]
    g = [0, 0, 0, 0]
    b = [0, 0, 0, 0]

    for i in range(0, int(x/2)):
        for j in range(0, int(y/2)):
            r[0], g[0], b[0] = img.getpixel((2 * i, 2 * j))
            r[1], g[1], b[1] = img.getpixel((2 * i + 1, 2 * j))
            r[2], g[2], b[2] = img.getpixel((2 * i, 2 * j + 1))
            r[3], g[3], b[3] = img.getpixel((2 * i + 1, 2 * j + 1))
            new_arr.putpixel((int(i), int(j)), (int((r[0] + r[1] + r[2] + r[3]) / 4), int(
                (g[0] + g[1] + g[2] + g[3]) / 4), int((b[0] + b[1] + b[2] + b[3]) / 4)))
    new_arr = np.uint8(new_arr)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_left():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 0), (0, 50)), 'constant')[:, 50:]
    g = np.pad(g, ((0, 0), (0, 50)), 'constant')[:, 50:]
    b = np.pad(b, ((0, 0), (0, 50)), 'constant')[:, 50:]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_right():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 0), (50, 0)), 'constant')[:, :-50]
    g = np.pad(g, ((0, 0), (50, 0)), 'constant')[:, :-50]
    b = np.pad(b, ((0, 0), (50, 0)), 'constant')[:, :-50]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_up():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 50), (0, 0)), 'constant')[50:, :]
    g = np.pad(g, ((0, 50), (0, 0)), 'constant')[50:, :]
    b = np.pad(b, ((0, 50), (0, 0)), 'constant')[50:, :]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_down():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    g = np.pad(g, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    b = np.pad(b, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_addition():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img).astype('uint16')
    img_arr = img_arr+100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_substraction():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img).astype('int16')
    img_arr = img_arr-100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_multiplication():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    img_arr = img_arr*1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_division():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    img_arr = img_arr/1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def convolution(img, kernel):
    h_img, w_img, _ = img.shape
    out = np.zeros((h_img-2, w_img-2), dtype=float)
    new_img = np.zeros((h_img-2, w_img-2, 3))
    if np.array_equal((img[:, :, 1], img[:, :, 0]), img[:, :, 2]) == True:
        array = img[:, :, 0]
        for h in range(h_img-2):
            for w in range(w_img-2):
                S = np.multiply(array[h:h+3, w:w+3], kernel)
                out[h, w] = np.sum(S)
        out_ = np.clip(out, 0, 255)
        for channel in range(3):
            new_img[:, :, channel] = out_
    else:
        for channel in range(3):
            array = img[:, :, channel]
            for h in range(h_img-2):
                for w in range(w_img-2):
                    S = np.multiply(array[h:h+3, w:w+3], kernel)
                    out[h, w] = np.sum(S)
            out_ = np.clip(out, 0, 255)
            new_img[:, :, channel] = out_
    new_img = np.uint8(new_img)
    return new_img


def edge_detection():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=float)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def blur():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=float) 
    kernel = np.array(
        [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def sharpening():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=float)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def histogram_rgb():
    img_path = "static/img/img_now.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    if is_grey_scale(img_path):
        g = img_arr[:, :, 0].flatten()
        data_g = Counter(g)
        plt.bar(list(data_g.keys()), data_g.values(), color='black')
        plt.savefig(f'static/img/grey_histogram.jpg', dpi=300)
        plt.clf()
    else:
        r = img_arr[:, :, 0].flatten()
        g = img_arr[:, :, 1].flatten()
        b = img_arr[:, :, 2].flatten()
        data_r = Counter(r)
        data_g = Counter(g)
        data_b = Counter(b)
        data_rgb = [data_r, data_g, data_b]
        warna = ['red', 'green', 'blue']
        data_hist = list(zip(warna, data_rgb))
        for data in data_hist:
            plt.bar(list(data[1].keys()), data[1].values(), color=f'{data[0]}')
            plt.savefig(f'static/img/{data[0]}_histogram.jpg', dpi=300)
            plt.clf()


def df(img):  # to make a histogram (count distribution frequency)
    values = [0]*256
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            values[img[i, j]] += 1
    return values


def cdf(hist):  # cumulative distribution frequency
    cdf = [0] * len(hist)  # len(hist) is 256
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i-1]+hist[i]
    # Now we normalize the histogram
    # What your function h was doing before
    cdf = [ele*255/cdf[-1] for ele in cdf]
    return cdf


def histogram_equalizer():
    img = cv2.imread('static\img\img_now.jpg', 0)
    my_cdf = cdf(df(img))
    # use linear interpolation of cdf to find new pixel values. Scipy alternative exists
    image_equalized = np.interp(img, range(0, 256), my_cdf)
    cv2.imwrite('static/img/img_now.jpg', image_equalized)

def threshold(lower_thres, upper_thres):
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img)  
    condition = np.logical_and(np.greater_equal(img_arr, lower_thres),
                               np.less_equal(img_arr, upper_thres))
    img_arr = img_arr.copy()  
    img_arr[condition] = 255
    new_img = Image.fromarray(img_arr)
    new_img.save("static/img/img_now.jpg")

def count_obj():
    img = Image.open("static/img/img_now.jpg").convert('L')
    img_arr = np.array(img)
    img_arr_copy = np.copy(img_arr)
    lower_white = np.array([200])
    upper_white = np.array([255])
    mask = cv2.inRange(img_arr, lower_white, upper_white)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    jumlah_obj = len(contours)
    result_image = cv2.cvtColor(img_arr_copy, cv2.COLOR_GRAY2BGR) 
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
    new_img = Image.fromarray(result_image)
    new_img.save("static/img/img_now.jpg")
    
    return jumlah_obj
    
def thinning(image):
    img_arr = np.array(image)  # Konversi gambar menjadi array numpy
    size = np.size(img_arr)
    skel = np.zeros(img_arr.shape, np.uint8)

    ret, img = cv2.threshold(img_arr, 127, 255, cv2.THRESH_BINARY_INV)  # Gunakan cv2.THRESH_BINARY_INV
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel

def get_freeman_chain_code_from_image(image):
    chain_code = []
    points = np.transpose(np.nonzero(image))
    directions = {
        (0, 1): 0,  # Arah 0
        (1, 1): 1,  # Arah 1
        (1, 0): 2,  # Arah 2
        (1, -1): 3,  # Arah 3
        (0, -1): 4,  # Arah 4
        (-1, -1): 5,  # Arah 5
        (-1, 0): 6,  # Arah 6
        (-1, 1): 7  # Arah 7
    }

    for i in range(1, len(points)):
        dx = points[i][1] - points[i - 1][1]
        dy = points[i][0] - points[i - 1][0]

        # Tentukan arah Freeman menggunakan lookup table
        direction = directions.get((dx, dy), None)
        if direction is not None:
            chain_code.append(direction)

    return chain_code

def count_obj_method2():
    img = Image.open("static/img/img_now.jpg").convert('L')
    img = img.resize((100, 100))
    img = thinning(img)
    img = Image.fromarray(img)
    img.save("static/img/img_now.jpg")
    chain_code = get_freeman_chain_code_from_image(img)
    
    # Menggunakan tanda kutip tunggal di awal dan di akhir string
    fcc_str = "'" + ",".join(map(str, chain_code)) + "'"

    # Simpan hasil Freeman Chain Code ke dalam file .env
    with open(".env", 'r') as f:
        env_lines = f.readlines()

    # Cari key yang sesuai untuk menyimpan string
    empty_key_index = None
    for i, line in enumerate(env_lines):
        if line.startswith(f"FCC_{i}="):
            if not line.strip().endswith('='):
                continue
            empty_key_index = i
            break

    if empty_key_index is None:
        empty_key_index = len(env_lines)

    # Simpan string Freeman Chain Code ke file .env dengan ' sebagai pemisah
    if empty_key_index >= len(env_lines):
        env_lines.append(f"FCC_{empty_key_index}={fcc_str}\n")
    else:
        env_lines[empty_key_index] = f"FCC_{empty_key_index}={fcc_str}\n"

    with open(".env", 'w') as f:
        f.writelines(env_lines)

    return fcc_str
def match_array_with_env():
    img = Image.open("static/img/img_now.jpg").convert('L')
    img = img.resize((100, 100))
    img = thinning(img)
    img = Image.fromarray(img)
    img.save("static/img/img_now.jpg")

    # Ubah hasil thinning menjadi Freeman Chain Code
    chain_code = get_freeman_chain_code_from_image(img)
    fcc_str = "'" + ",".join(map(str, chain_code)) + "'"  # Tambahkan tanda kutip di awal dan akhir

    # Baca setiap baris dari file .env
    with open(".env", 'r') as f:
        env_lines = f.readlines()

    # Fungsi untuk memisahkan string yang diapit oleh tanda kutip tunggal
    def split_by_quotes(s):
        parts = s.split("'")
        return [part for part in parts if part]

    # Inisialisasi variabel untuk menyimpan hasil pencocokan dan indeks baris
    matched_indices = []
    result_str = ""
    line = "1"
    
    for i, line1 in enumerate(env_lines):
        if line1.startswith("FCC_") and "=" in line1:
            stored_fcc_str = line1.strip().split('=', 1)[1]
            stored_fcc_sections = split_by_quotes(stored_fcc_str)

            distance = Levenshtein.distance(fcc_str, stored_fcc_str)
            
            if distance == 0:  # Jika Freeman Chain Code cocok
                matched_indices.append(i)
                result_str += str(i)
                line = i+1
                break  # Hentikan pencarian setelah menemukan emoji yang cocok
            

    emoji_dict = {
        '0': 'Blush',
        '1': 'Disappointed relieved',
        '2': 'Expressionless',
        '3': 'Face with raised eyebrow',
        '4': 'Face with rolling eyes',
        '5': 'Grin',
        '6': 'Grinning',
        '7': 'Heart eyes',
        '8': 'Hugging face',
        '9': 'Hushed',
        '10': 'Joy',
        '11': 'Kissing',
        '12': 'Kissing closed eyes',
        '13': 'Kissing heart',
        '14': 'Kissing smiling eyes',
        '15': 'Laughing',
        '16': 'Neutral face',
        '17': 'No mouth',
        '18': 'Open mouth',
        '19': 'Persevere',
        '20': 'Relaxed',
        '21': 'Rolling on the floor laughing',
        '22': 'Sleeping',
        '23': 'Sleepy',
        '24': 'Slightly smiling face',
        '25': 'Smile',
        '26': 'Smiley',
        '27': 'Smirk',
        '28': 'Star-struck',
        '29': 'Sunglasses',
        '30': 'Sweat smile',
        '31': 'Thinking face',
        '32': 'Tired face',
        '33': 'Wink',
        '34': 'Yum',
        '35': 'Zipper mouth face'
    }

    if result_str in emoji_dict:
        emoji_name = emoji_dict[result_str]
        return f"Emoji yang cocok ditemukan: '{emoji_name}' berdasarkan line {line} pada .env"
    else:
        return "Tidak ada emoji yang cocok ditemukan."

# Fungsi split_by_quotes tidak diperlukan karena tidak ada pembagian section
