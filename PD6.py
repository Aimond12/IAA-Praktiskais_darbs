import cv2
import numpy as np
import matplotlib.pyplot as plt


def convert_to_grayscale(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def canny_operator(gray, params):
    gray_filtered = cv2.GaussianBlur(
        gray,
        params['canny_gaussian_kernel'],
        params['canny_gaussian_sigma']
    )

    return cv2.Canny(
        gray_filtered,
        params['canny_threshold_1'],
        params['canny_threshold_2']
    )


def dilate_mask(mask, params):
    kernel = np.ones(params['dilation_kernel'], np.uint8)

    return cv2.dilate(
        mask,
        kernel,
        iterations=params['dilation_iterations']
    )


def mean_filter(gray, params):
    return cv2.blur(
        gray,
        params['mean_kernel']
    )

def weak_mean_filter(gray, params):
    return cv2.blur(
        gray,
        params['weak_mean_kernel']
    )

def combine_by_mask(edge_image, blurred_image, mask):
    return np.where(mask == 255, edge_image, blurred_image).astype(np.uint8)


def combined_algorithm(image, params):
    gray = convert_to_grayscale(image)

    edge_mask = canny_operator(gray, params)

    dilated_mask = dilate_mask(edge_mask, params)

    weak_mean = weak_mean_filter(gray, params)

    strong_mean = mean_filter(gray, params)

    final = combine_by_mask(
        weak_mean,
        strong_mean,
        dilated_mask
    )

    return {
        'gray': gray,
        'canny': edge_mask,
        'mask': dilated_mask,
        'weak_mean': weak_mean,
        'strong_mean': strong_mean,
        'final': final
    }


def show_one_row(item):
    result = combined_algorithm(item['original'], item['params'])

    imgs = [
        result['gray'],
        result['canny'],
        result['mask'],
        result['weak_mean'],
        result['strong_mean'],
        result['final']
    ]

    row_titles = [
        'Original',
        'Edges',
        'Dilated Edge Mask',
        'Weak Mean Filter',
        'Strong Mean Filter',
        'Final Result'
    ]

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))

    for j, img in enumerate(imgs):
        axes[j].imshow(img, cmap='gray')
        axes[j].set_title(row_titles[j])
        axes[j].axis('off')

    p = item['params']

    plt.suptitle(
        f"{item['name']} - "
        f"Canny: {p['canny_threshold_1']}/{p['canny_threshold_2']}, "
        f"Mean: {p['mean_kernel']}",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()


image_paths = [
    'image1.jpg',
    'image2.jpg',
    'image3.jpg',
    'image4.jpg',
    'image5.jpg',
]


image_params = [
    {
        'canny_gaussian_kernel': (5, 5),
        'canny_gaussian_sigma': 1.4,
        'canny_threshold_1': 35,
        'canny_threshold_2': 90,
        'dilation_kernel': (3, 3),
        'dilation_iterations': 1,
        'weak_mean_kernel' : (3,3),
        'mean_kernel': (11, 11)
    },
    {
        'canny_gaussian_kernel': (15, 15),
        'canny_gaussian_sigma': 2,
        'canny_threshold_1': 25,
        'canny_threshold_2': 70,
        'dilation_kernel': (3, 3),
        'dilation_iterations': 1,
        'weak_mean_kernel' : (3,3),
        'mean_kernel': (7, 7)
    },
    {
        'canny_gaussian_kernel': (15, 15),
        'canny_gaussian_sigma': 1.7,
        'canny_threshold_1': 60,
        'canny_threshold_2': 120,
        'dilation_kernel': (3, 3),
        'dilation_iterations': 1,
        'weak_mean_kernel' : (3,3),        
        'mean_kernel': (7, 7)
    },
    {
        'canny_gaussian_kernel': (11, 11),
        'canny_gaussian_sigma': 2.4,
        'canny_threshold_1': 0,
        'canny_threshold_2': 50,
        'dilation_kernel': (5, 5),
        'dilation_iterations': 1,
        'weak_mean_kernel' : (3,3),  
        'mean_kernel': (11, 11)
    },
    {
        'canny_gaussian_kernel': (7, 7),
        'canny_gaussian_sigma': 1.4,
        'canny_threshold_1': 60,
        'canny_threshold_2': 130,
        'dilation_kernel': (3, 3),
        'dilation_iterations': 1,
        'weak_mean_kernel' : (3,3),  
        'mean_kernel': (11, 11)
    }
]


images = []

for p in image_paths:
    img = cv2.imread(p)

    if img is None:
        print(f"Could not load image: {p}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)


dataset = []

for i, img in enumerate(images):
    dataset.append({
        'name': f'Image {i + 1}',
        'original': img,
        'params': image_params[i]
    })

show_one_row(dataset[0])
show_one_row(dataset[1])
show_one_row(dataset[2])
show_one_row(dataset[3])
show_one_row(dataset[4])
