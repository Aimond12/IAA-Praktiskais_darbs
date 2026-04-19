import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img1 = load_image('image1.jpg')
img3 = load_image('image2.jpg')

def add_gaussian_noise(image, mean=0, sigma=30):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

img2 = add_gaussian_noise(img1)

def apply_canny(image, blur_ksize=9):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_filtered = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1.4)
    edges = cv2.Canny(gray_filtered, 50, 150) 
    return edges

def roberts_operator(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32)

    h, w = gray.shape
    gradient = np.zeros((h, w))

    for y in range(h - 1):
        for x in range(w - 1):
            a = gray[y, x]
            b = gray[y, x+1]
            c = gray[y+1, x]
            d = gray[y+1, x+1]

            Gx = a - d
            Gy = b - c

            gradient[y, x] = np.sqrt(Gx**2 + Gy**2)
    max_val = gradient.max()
    if max_val > 0:
        gradient = (gradient / max_val * 255)
    binary = np.where(gradient >= threshold, 255, 0).astype(np.uint8)
    return binary

images = [img1, img2, img3]
titles = ["Original", "Noisy", "Custom"]

img1_canny = apply_canny(img1)
img2_canny = apply_canny(img2)
img3_canny = apply_canny(img3)

img1_roberts = roberts_operator(img1, threshold=60)
img2_roberts = roberts_operator(img2, threshold=100)
img3_roberts = roberts_operator(img3, threshold=20)

canny_results = [img1_canny, img2_canny, img3_canny]
roberts_results = [img1_roberts, img2_roberts, img3_roberts]

def show_results():
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))

    for i in range(3):
        axes[i, 0].imshow(images[i], cmap='gray')
        axes[i, 0].set_title(f"{titles[i]} Image")

        axes[i, 1].imshow(canny_results[i], cmap='gray')
        axes[i, 1].set_title("Canny")

        axes[i, 2].imshow(roberts_results[i], cmap='gray')
        axes[i, 2].set_title("Roberts")

        for j in range(3):
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()

show_results()
