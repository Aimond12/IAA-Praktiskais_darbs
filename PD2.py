import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(filename):

    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
img1 = read_image("dark.png")
img2 = read_image("bright.png")
img3 = read_image("lowcontrast.png")

def log_correction(img):

    img_float = img.astype(np.float32)

    img_float = img_float / 255.0

    c = 1 / np.log(1 + np.max(img_float))
    log_img = c * np.log(1 + img_float)

    log_img = np.uint8(log_img * 255)

    return log_img

def linear_contrast(img):

    result = np.zeros_like(img, dtype=np.float32)

    for i in range(3):

        channel = img[:,:,i].astype(np.float32)

        Imin = np.min(channel)
        Imax = np.max(channel)

        new_channel = 255 * (channel - Imin) / (Imax - Imin + 1e-6)

        result[:,:,i] = new_channel

    return np.clip(result,0,255).astype(np.uint8)



def plot_histogram(img, title):

    colors = ('r','g','b')

    plt.title(title)

    for i,c in enumerate(colors):

        hist = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(hist,color=c)

    plt.xlim([0,256])

def process(img, title):

    log_img = log_correction(img)
    lin_img = linear_contrast(img)

    plt.figure(figsize=(15,10))
    plt.suptitle(title, fontsize=16)

    plt.subplot(2,3,1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2,3,2)
    plt.imshow(log_img)
    plt.title("Log correction")
    plt.axis("off")

    plt.subplot(2,3,3)
    plt.imshow(lin_img)
    plt.title("Linear contrast")
    plt.axis("off")

    plt.subplot(2,3,4)
    plot_histogram(img,"Original histogram")

    plt.subplot(2,3,5)
    plot_histogram(log_img,"Log histogram")

    plt.subplot(2,3,6)
    plot_histogram(lin_img,"Linear histogram")

    plt.tight_layout()
    plt.show()


process(img1, "Pārtumšots attēls (Dark image)")
process(img2, "Pārgaismots attēls (Bright image)")
process(img3, "Pelēcīgs attēls (Low contrast image)")
