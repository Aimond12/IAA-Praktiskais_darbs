import cv2
import numpy as np
import matplotlib.pyplot as plt

image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]


def adaptive_threshold(img):
    return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,5
    )
def region_growing(img, seed, threshold):
    h, w = img.shape
    segmented = np.zeros((h, w), dtype=np.uint8)

    seed_value = img[seed]
    stack = [seed]

    while stack:
        x, y = stack.pop()

        if segmented[x, y] == 0:
            segmented[x, y] = 255

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy

                    if 0 <= nx < h and 0 <= ny < w:
                        if segmented[nx, ny] == 0:
                            if abs(int(img[nx, ny]) - int(seed_value)) < threshold:
                                stack.append((nx, ny))

    return segmented

def show_images(images, titles):
    plt.figure(figsize=(15, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

for path in image_paths:

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = adaptive_threshold(gray)

    h, w = gray.shape
    seed = (h // 2, w // 2)
    region = region_growing(gray, seed, threshold=70)

    show_images(
        [gray, thresh, region],
        ["Original", "Adaptive Threshold", "Region Growing"]
    )
