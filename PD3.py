import cv2
import numpy as np
import matplotlib.pyplot as plt


def add_gaussian_noise(image, mean=0, sigma=55):
    image_float = image.astype(np.float32)
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image_float + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_jpeg_artifacts(image, quality=8):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg


def show_one_row(item):
    noisy = item['noisy']

    mean_f = cv2.blur(noisy, (11, 11))
    median_f = cv2.medianBlur(noisy, 9)
    gauss_f = cv2.GaussianBlur(noisy, (13, 13), 0)

    imgs = [item['original'], noisy, mean_f, median_f, gauss_f]

    row_titles = [
        'Original',
        f"Noisy ({item['type']})",
        'Mean Filter',
        'Median Filter',
        'Gaussian Filter'
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for j, img in enumerate(imgs):
        axes[j].imshow(img)
        axes[j].set_title(row_titles[j])
        axes[j].axis('off')

    plt.suptitle(f"{item['name']} - {item['type']}", fontsize=14)
    plt.tight_layout()
    plt.show()
image_paths = ['image1.jpg', 'image2.jpg']
images = []

for p in image_paths:
    img = cv2.imread(p)
    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

dataset = []
for i, img in enumerate(images):
    dataset.append({
        'name': f'Image {i+1}',
        'type': 'Gaussian',
        'original': img,
        'noisy': add_gaussian_noise(img)
    })
    dataset.append({
        'name': f'Image {i+1}',
        'type': 'JPEG',
        'original': img,
        'noisy': add_jpeg_artifacts(img)
    })
show_one_row(dataset[0])
show_one_row(dataset[1])
show_one_row(dataset[2])
show_one_row(dataset[3])
