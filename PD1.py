import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

img1 = cv2.imread("image1.jpg").astype(np.float32) / 255.0
img2 = cv2.imread("image2.jpg").astype(np.float32) / 255.0

if img1.shape != img2.shape:
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

def multiply_blend(A, B):
    return A * B

def screen_blend(A, B):
    return 1 - (1 - A) * (1 - B)

def color_burn_blend(A, B):
    epsilon = 1e-6
    A_safe = np.clip(A, epsilon, 1.0)
    result = 1 - (1 - B) / A_safe
    return np.clip(result, 0, 1)

def linear_dodge_blend(A, B):
    return np.clip(A + B, 0, 1)

multiply = multiply_blend(img1, img2)
screen = screen_blend(img1, img2)
burn = color_burn_blend(img1, img2)
dodge = linear_dodge_blend(img1, img2)

def show_row(title, A, B, C):
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    images = [A, B, C]
    titles = ["Image A", "Image B", title]
    for ax, img, t in zip(axes, images, titles):
        ax.imshow(cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax.set_title(t)
        ax.axis("off")
    plt.show()

show_row("Multiply", img1, img2, multiply)
show_row("Screen", img1, img2, screen)
show_row("Color Burn", img1, img2, burn)
show_row("Linear Dodge", img1, img2, dodge)
