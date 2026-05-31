import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def add_gaussian_noise(image, mean, sigma):
  image_float = image.astype(np.float32)
  noise = np.random.normal(mean, sigma, image.shape)
  noisy = image_float + noise
  noisy = np.clip(noisy, 0, 255).astype(np.uint8)
  return noisy

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

def calculate_mse(original, processed):
    original = original.astype(np.float32)
    processed = processed.astype(np.float32)

    return np.mean((original - processed) ** 2)

def calculate_ssim(original, processed):
    return ssim(original, processed, data_range=255)

def calculate_metrics(original_gray, noisy_gray, processed_gray):
    noisy_metrics = {
        'MSE': calculate_mse(original_gray, noisy_gray),
        'SSIM': calculate_ssim(original_gray, noisy_gray)
    }

    processed_metrics = {
        'MSE': calculate_mse(original_gray, processed_gray),
        'SSIM': calculate_ssim(original_gray, processed_gray)
    }

    return noisy_metrics, processed_metrics

def show_result(item):
    original = item['original']
    params = item['params']

    noisy = add_gaussian_noise(
        original,
        mean=params['noise_mean'],
        sigma=params['noise_sigma']
    )


    original_gray = convert_to_grayscale(original)

    noisy_gray = convert_to_grayscale(noisy)
    processed = combined_algorithm(noisy, params)['final']

    noisy_metrics, processed_metrics = calculate_metrics(
        original_gray,
        noisy_gray,
        processed
    )

    images = [
        original_gray,
        noisy_gray,
        processed
    ]

    titles = [
        'Original',
        'Noisy',
        'Processed'
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    for i, img in enumerate(images):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(titles[i])
        axes[i].axis('off')

    plt.suptitle(
        f"{item['name']} | "
        f"Canny: {params['canny_threshold_1']}/{params['canny_threshold_2']} | "
        f"Mean: {params['mean_kernel']}",
        fontsize=13
    )

    plt.tight_layout()
    plt.show()

    print(f"\n{item['name']} metrics")
    print("-" * 60)

    print("Noisy image vs Original:")
    print(f"MSE:   {noisy_metrics['MSE']:.2f}")
    print(f"SSIM:  {noisy_metrics['SSIM']:.4f}")

    print("\nProcessed image vs Original:")
    print(f"MSE:   {processed_metrics['MSE']:.2f}")
    print(f"SSIM:  {processed_metrics['SSIM']:.4f}")


image_paths = [
    'image1.jpeg',
    'image2.jpg',
    'image3.jpeg'
]


image_params = [
    {
        'noise_mean': 0,
        'noise_sigma': 35,
                
        'canny_gaussian_kernel': (15, 15),
        'canny_gaussian_sigma': 1.7,
        'canny_threshold_1': 35,
        'canny_threshold_2': 135,
        'dilation_kernel': (3, 3),
        'dilation_iterations': 2,
        'weak_mean_kernel' : (3,3),
        'mean_kernel': (11, 11)
    },
    {
        'noise_mean': 0,
        'noise_sigma': 35,
                
        'canny_gaussian_kernel': (13, 13),
        'canny_gaussian_sigma': 1.6,
        'canny_threshold_1': 40,
        'canny_threshold_2': 95,
        'dilation_kernel': (7, 7),
        'dilation_iterations': 1,
        'weak_mean_kernel' : (3,3),
        'mean_kernel': (15, 15)

    },
        {
        'noise_mean': 0,
        'noise_sigma': 65,
                    
        'canny_gaussian_kernel': (13, 13),
        'canny_gaussian_sigma': 1.8,
        'canny_threshold_1': 70,
        'canny_threshold_2': 95,
        'dilation_kernel': (5, 5),
        'dilation_iterations': 2,
        'weak_mean_kernel' : (3,3),
        'mean_kernel': (13, 13)

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

show_result(dataset[0])
show_result(dataset[1])
show_result(dataset[2])
