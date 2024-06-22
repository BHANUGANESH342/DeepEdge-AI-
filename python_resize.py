import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to load image at {image_path}")
    return img

def resize_with_interpolation(img, output_size):
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(cv2.resize, img, output_size, interpolation=cv2.INTER_NEAREST): 'INTER_NEAREST',
            executor.submit(cv2.resize, img, output_size, interpolation=cv2.INTER_LINEAR): 'INTER_LINEAR',
            executor.submit(cv2.resize, img, output_size, interpolation=cv2.INTER_CUBIC): 'INTER_CUBIC',
        }

        results = {}
        for future in as_completed(futures):
            method = futures[future]
            results[method] = future.result()

    return results['INTER_NEAREST'], results['INTER_LINEAR'], results['INTER_CUBIC']

def measure_timing(img, output_size):
    timings = {}
    def measure_single_method(interpolation_flag):
        start_time = time.time()
        for _ in range(1000):
            _ = cv2.resize(img, output_size, interpolation=interpolation_flag)
        end_time = time.time()
        return (end_time - start_time) * 1000  # convert to milliseconds
    for interpolation_flag in [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]:
        timings[interpolation_flag] = measure_single_method(interpolation_flag)
    return timings

def custom_resize(img, output_size):
    img_height, img_width = img.shape[:2]
    new_height, new_width = output_size
    new_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    for y in range(new_height):
        for x in range(new_width):
            src_y = int(y * img_height / new_height)
            src_x = int(x * img_width / new_width)
            new_img[y, x] = img[src_y, src_x]
    return new_img

def display_resized_images(resized_nearest, resized_linear, resized_cubic, custom_resized):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Resized Images Using Different Interpolation Methods')
    axes[0, 0].imshow(cv2.cvtColor(resized_nearest, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('INTER_NEAREST')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(cv2.cvtColor(resized_linear, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('INTER_LINEAR')
    axes[0, 1].axis('off')
    axes[1, 0].imshow(cv2.cvtColor(resized_cubic, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('INTER_CUBIC')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(cv2.cvtColor(custom_resized, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Custom Resized')
    axes[1, 1].axis('off')
    plt.show()

# Function to plot 3D image with differences highlighted in red
def plot_3d_image(original_img, resized_img, title, downscale_factor=4):
    resized_original = cv2.resize(original_img, (resized_img.shape[1], resized_img.shape[0]))
    original_gray = cv2.cvtColor(resized_original, cv2.COLOR_BGR2GRAY)
    resized_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    diff = np.abs(original_gray - resized_gray)

    # Downsample the image and the difference for faster plotting
    original_gray = cv2.resize(original_gray, (original_gray.shape[1] // downscale_factor, original_gray.shape[0] // downscale_factor))
    resized_gray = cv2.resize(resized_gray, (resized_gray.shape[1] // downscale_factor, resized_gray.shape[0] // downscale_factor))
    diff = cv2.resize(diff, (diff.shape[1] // downscale_factor, diff.shape[0] // downscale_factor))

    x = np.arange(resized_gray.shape[1])
    y = np.arange(resized_gray.shape[0])
    x, y = np.meshgrid(x, y)
    z = resized_gray

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, facecolors=plt.cm.gray(z / 255), rstride=1, cstride=1, antialiased=False, shade=False)
    ax.contourf(x, y, diff, zdir='z', offset=z.min() - 20, cmap='Reds', alpha=0.6)
    ax.set_title(title)
    plt.show()

if __name__ == "__main__":
    input_image_path = r"D:\bhanu\desktop\Projects_working\veipvev\G178_2 -1080.BMP"
    try:

        img = load_image(input_image_path)

        output_size = (int(img.shape[1] / 2), int(img.shape[0] / 2))

        resized_nearest, resized_linear, resized_cubic = resize_with_interpolation(img, output_size)

        timings = measure_timing(img, output_size)
        print(f"Time taken for 1000 iterations using INTER_NEAREST: {timings[cv2.INTER_NEAREST]:.2f} ms")
        print(f"Time taken for 1000 iterations using INTER_LINEAR: {timings[cv2.INTER_LINEAR]:.2f} ms")
        print(f"Time taken for 1000 iterations using INTER_CUBIC: {timings[cv2.INTER_CUBIC]:.2f} ms")

        custom_resized = custom_resize(img, output_size)

        display_resized_images(resized_nearest, resized_linear, resized_cubic, custom_resized)

        labels = ['INTER_NEAREST', 'INTER_LINEAR', 'INTER_CUBIC']
        times = [timings[cv2.INTER_NEAREST], timings[cv2.INTER_LINEAR], timings[cv2.INTER_CUBIC]]
        plt.bar(labels, times, color=['blue', 'green', 'red'])
        plt.xlabel('Interpolation Method')
        plt.ylabel('Time (ms)')
        plt.title('Timing for 1000 Iterations of Different Interpolation Methods')
        plt.show()
        # Plot 3D images
        plot_3d_image(img, resized_nearest, 'INTER_NEAREST')
        plot_3d_image(img, resized_linear, 'INTER_LINEAR')
        plot_3d_image(img, resized_cubic, 'INTER_CUBIC')
        plot_3d_image(img, custom_resized, 'Custom Resized')
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
