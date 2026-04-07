import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_image_rgb(path):
    image_bgr = cv2.imread(path)
    if image_bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

def show_images(images, titles, cols=2, figsize=(12, 6)):
    rows = int(np.ceil(len(images) / cols))
    plt.figure(figsize=figsize)

    for i, (img, title) in enumerate(zip(images, titles), 1):
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.title(title, fontsize=12)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def add_gaussian_noise(image, mean=0, sigma=55):
    image_float = image.astype(np.float32)
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image_float + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def add_jpeg_artifacts(image, quality=8):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    success, encimg = cv2.imencode(".jpg", image_bgr, encode_param)
    if not success:
        raise ValueError("JPEG encoding failed")

    decoded_bgr = cv2.imdecode(encimg, 1)
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    return decoded_rgb

def show_noisy_images(dataset, image_paths):
    gaussian_row = []
    jpeg_row = []

    for image_name in image_paths:
        for item in dataset:
            if item["image_name"] == image_name and item["noise_type"] == "Gaussian noise":
                gaussian_row.append(item["noisy"])
            if item["image_name"] == image_name and item["noise_type"] == "JPEG artifacts":
                jpeg_row.append(item["noisy"])

    noisy_images = gaussian_row + jpeg_row
    noisy_titles = [
        f"{image_paths[0]} - Gaussian noise",
        f"{image_paths[1]} - Gaussian noise",
        f"{image_paths[0]} - JPEG artifacts",
        f"{image_paths[1]} - JPEG artifacts"
    ]

    print("Noisy images:")
    show_images(noisy_images, noisy_titles, cols=2, figsize=(12, 10))



def build_results(dataset):
    results = []

    for item in dataset:
        original = item["original"]
        noisy = item["noisy"]

        mean_filtered = cv2.blur(noisy, (11, 11))
        median_filtered = cv2.medianBlur(noisy, 9)
        gaussian_filtered = cv2.GaussianBlur(noisy, (13, 13), 0)

        results.append({
            "image_name": item["image_name"],
            "noise_type": item["noise_type"],
            "original": original,
            "noisy": noisy,
            "mean": mean_filtered,
            "median": median_filtered,
            "gaussian": gaussian_filtered
        })

    return results



def get_result(results, image_name, noise_type):
    for item in results:
        if item["image_name"] == image_name and item["noise_type"] == noise_type:
            return item
    raise ValueError(f"Result not found for {image_name} and {noise_type}")


def show_filter_comparison_for_image(results, image_name, figsize=(12, 24)):
    gaussian_item = get_result(results, image_name, "Gaussian noise")
    jpeg_item = get_result(results, image_name, "JPEG artifacts")

    comparisons = [
        {
            "row_label": "Gaussian noise → Mean filter",
            "filter_title": "Mean filter result",
            "noisy": gaussian_item["noisy"],
            "filtered": gaussian_item["mean"]
        },
        {
            "row_label": "Gaussian noise → Median filter",
            "filter_title": "Median filter result",
            "noisy": gaussian_item["noisy"],
            "filtered": gaussian_item["median"]
        },
        {
            "row_label": "Gaussian noise → Gaussian filter",
            "filter_title": "Gaussian filter result",
            "noisy": gaussian_item["noisy"],
            "filtered": gaussian_item["gaussian"]
        },
        {
            "row_label": "JPEG artifacts → Mean filter",
            "filter_title": "Mean filter result",
            "noisy": jpeg_item["noisy"],
            "filtered": jpeg_item["mean"]
        },
        {
            "row_label": "JPEG artifacts → Median filter",
            "filter_title": "Median filter result",
            "noisy": jpeg_item["noisy"],
            "filtered": jpeg_item["median"]
        },
        {
            "row_label": "JPEG artifacts → Gaussian filter",
            "filter_title": "Gaussian filter result",
            "noisy": jpeg_item["noisy"],
            "filtered": jpeg_item["gaussian"]
        }
    ]

    fig, axes = plt.subplots(6, 2, figsize=figsize)

    for row, item in enumerate(comparisons):
       
        axes[row, 0].imshow(item["noisy"])
        axes[row, 0].axis("off")
        axes[row, 0].set_title("Noisy image", fontsize=12)

        
        axes[row, 1].imshow(item["filtered"])
        axes[row, 1].axis("off")
        axes[row, 1].set_title(item["filter_title"], fontsize=12)

      
        axes[row, 0].set_ylabel(
            item["row_label"],
            fontsize=11,
            rotation=90,
            labelpad=30,
            fontweight="bold"
        )

    plt.suptitle(f"Filter comparison for {image_name}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


image_paths = ["image1.jpg", "image2.jpg"]

images = [read_image_rgb(path) for path in image_paths]

print("Original images:")
show_images(images, [f"Original: {name}" for name in image_paths], cols=2, figsize=(12, 6))


dataset = []

for name, img in zip(image_paths, images):
    gaussian_noisy = add_gaussian_noise(img)
    jpeg_noisy = add_jpeg_artifacts(img)

    dataset.append({
        "image_name": name,
        "noise_type": "Gaussian noise",
        "original": img,
        "noisy": gaussian_noisy
    })

    dataset.append({
        "image_name": name,
        "noise_type": "JPEG artifacts",
        "original": img,
        "noisy": jpeg_noisy
    })


show_noisy_images(dataset, image_paths)

results = build_results(dataset)

show_filter_comparison_for_image(results, image_paths[0], figsize=(12, 24))

show_filter_comparison_for_image(results, image_paths[1], figsize=(12, 24))
