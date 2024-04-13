
'''
LOAD AND DISPLAY

from PIL import Image

# Load the image
image_path = 'E:/nature.jpg'  # Replace with the path to your image file
image = Image.open(image_path)

# Display the image
image.show()
'''





'''
CREATE RANDOM ARRAY AND STORE

import numpy as np
from PIL import Image

# Define the size of the array
width, height = 100, 100  # You can adjust the width and height as needed

# Create a random array with values in the range 0-255 (for 8-bit images)
random_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

# Convert the array to a Pillow Image
image = Image.fromarray(random_array)

# Save the image to a file
image_path = 'E:/random.jpg'  # Define the output file path and name
image.save(image_path)

print(f'Random image saved at {image_path}')

'''










'''
RESIZING AN IMAGE

from PIL import Image

# Load the image
image_path = 'E:/nature.jpg'  # Replace with the path to your image file
image = Image.open(image_path)

# Define the new size for the image
new_width, new_height = 200, 200  # You can adjust the width and height as needed

# Resize the image
resized_image = image.resize((new_width, new_height))

# Save the resized image to a new file
resized_image_path = 'E:/resized_image.jpg'
  # Define the output file path and name
resized_image.save(resized_image_path)

print(f'Resized image saved at {resized_image_path}')

'''






'''

ROTATING AN IMAGE

from PIL import Image

# Load the image
image_path = 'E:/nature.jpg'  # Replace with the path to your image file
image = Image.open(image_path)

# Define the rotation angle in degrees (counter-clockwise)
rotation_angle = 45  # You can specify any angle you want

# Rotate the image
# By default, expand is set to False. Set it to True if you want the output image to be expanded to fit the rotation.
rotated_image = image.rotate(rotation_angle, expand=True)

# Save the rotated image to a new file
rotated_image_path = 'E:/rotated_image.jpg'  # Define the output file path and name
rotated_image.save(rotated_image_path)

print(f'Rotated image saved at {rotated_image_path}')
'''






'''
UNIFORM QUANTIZATION 4 8 16 32 64

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image_path = 'E:/nature.jpg'  # Replace with the path to your image file
image = Image.open(image_path)

# Convert the image to grayscale
gray_image = image.convert('L')

# Convert the grayscale image to a numpy array
gray_array = np.array(gray_image)

# Function to quantize the image to a specified number of gray levels
def quantize_image(image_array, num_levels):
    # Calculate the interval between quantization levels
    interval = 256 // num_levels
    # Calculate the quantization thresholds
    thresholds = np.linspace(0, 255, num_levels + 1)
    
    # Apply uniform quantization to the image
    quantized_image = np.digitize(image_array, thresholds, right=True) - 1
    quantized_image = quantized_image * interval
    
    return quantized_image

# Apply uniform quantization for 4, 8, 16, 32, and 64 gray levels
levels = [4, 8, 16, 32, 64]

# Plot original and quantized images
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(gray_array, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

for i, level in enumerate(levels):
    # Apply quantization
    quantized_array = quantize_image(gray_array, level)
    
    # Convert the quantized array back to an image
    quantized_image = Image.fromarray(quantized_array.astype(np.uint8))
    
    # Display the quantized image
    ax = axes[(i + 1) // 3, (i + 1) % 3]
    ax.imshow(quantized_image, cmap='gray')
    ax.set_title(f'Quantized to {level} gray levels')
    ax.axis('off')
    
    # Save the quantized image if desired
    # quantized_image.save(f'quantized_{level}_gray_levels.jpg')

plt.show()
'''









'''

GENERATE HISTOGRAM FOR ANY IMAGE AND DISPLAY

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image_path = 'E:/nature.jpg'  # Replace with the path to your image file
image = Image.open(image_path)

# Convert the image to grayscale for simplicity
gray_image = image.convert('L')

# Convert the grayscale image to a numpy array
gray_array = np.array(gray_image)

# Generate histogram data for the image
histogram, bins = np.histogram(gray_array, bins=256, range=(0, 256))

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.bar(bins[:-1], histogram, width=1, color='gray', alpha=0.7)
plt.title('Histogram of the Image')
plt.xlabel('Pixel Intensity (Gray Level)')
plt.ylabel('Frequency')
plt.show()

# Analyze the histogram
# Analysis can include examining the shape of the histogram:
# - If the histogram is concentrated on the left, the image is dark.
# - If the histogram is concentrated on the right, the image is bright.
# - If the histogram is spread evenly, the image has a wide range of brightness levels.
# - If the histogram is very narrow, the image may have low contrast.
'''





'''

HISTOGRAM EQUALIZATION

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image_path = 'E:/nature.jpg'  # Replace with the path to your image file
image = Image.open(image_path)

# Convert the image to grayscale
gray_image = image.convert('L')

# Convert the grayscale image to a numpy array
gray_array = np.array(gray_image)

# Function to perform histogram equalization
def histogram_equalization(image_array):
    # Calculate the histogram of the image
    histogram, bins = np.histogram(image_array, bins=256, range=(0, 256))
    
    # Calculate the cumulative distribution function (CDF)
    cdf = np.cumsum(histogram)
    cdf = cdf / cdf[-1]  # Normalize the CDF
    
    # Create a mapping from original gray levels to equalized gray levels
    equalized_mapping = np.round(cdf * 255).astype(np.uint8)
    
    # Apply the mapping to the original image array
    equalized_array = equalized_mapping[image_array]
    
    return equalized_array

# Perform histogram equalization
equalized_array = histogram_equalization(gray_array)

# Convert the equalized array back to an image
equalized_image = Image.fromarray(equalized_array)

# Plot the histograms before and after equalization
plt.figure(figsize=(12, 6))

# Histogram before equalization
plt.subplot(1, 2, 1)
plt.hist(gray_array.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
plt.title('Histogram Before Equalization')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

# Histogram after equalization
plt.subplot(1, 2, 2)
plt.hist(equalized_array.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
plt.title('Histogram After Equalization')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Display the original and equalized images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

plt.tight_layout()
plt.show()

'''







'''

CONTRAST STRETCHING

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image_path = 'E:/nature.jpg'  # Replace with the path to your image file
image = Image.open(image_path)

# Convert the image to grayscale
gray_image = image.convert('L')

# Convert the grayscale image to a numpy array
gray_array = np.array(gray_image)

# Function to perform contrast stretching
def contrast_stretching(image_array):
    # Find the minimum and maximum pixel intensity values in the image
    min_val = image_array.min()
    max_val = image_array.max()
    
    # Apply contrast stretching
    # Formula: (pixel - min) * (255 / (max - min))
    stretched_array = ((image_array - min_val) * (255 / (max_val - min_val))).astype(np.uint8)
    
    return stretched_array

# Perform contrast stretching
stretched_array = contrast_stretching(gray_array)

# Convert the stretched array back to an image
stretched_image = Image.fromarray(stretched_array)

# Plot the histograms before and after contrast stretching
plt.figure(figsize=(12, 6))

# Histogram before contrast stretching
plt.subplot(1, 2, 1)
plt.hist(gray_array.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
plt.title('Histogram Before Contrast Stretching')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

# Histogram after contrast stretching
plt.subplot(1, 2, 2)
plt.hist(stretched_array.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
plt.title('Histogram After Contrast Stretching')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Display the original and stretched images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(stretched_image, cmap='gray')
plt.title('Contrast Stretched Image')
plt.axis('off')

plt.tight_layout()
plt.show()


'''






'''

# ZERO PADDING

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image_path = 'E:/nature.jpg'  # Replace with the path to your image file
image = Image.open(image_path)

# Convert the image to grayscale
gray_image = image.convert('L')

# Convert the grayscale image to a numpy array
gray_array = np.array(gray_image)

# Define the amount of padding
padding_width = 10  # Change this value as needed

# Perform zero padding
zero_padded_array = np.pad(gray_array, ((padding_width, padding_width), (padding_width, padding_width)), mode='constant', constant_values=0)

# Convert the zero-padded array back to an image
zero_padded_image = Image.fromarray(zero_padded_array)

# Display the original and zero-padded images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(zero_padded_image, cmap='gray')
plt.title('Zero Padded Image')
plt.axis('off')

plt.tight_layout()
plt.show()



# CONSTANT PADDING

# Define the padding value and padding width
padding_value = 100  # Replace with your desired constant value
padding_width = 10  # Change this value as needed

# Perform constant padding
constant_padded_array = np.pad(gray_array, ((padding_width, padding_width), (padding_width, padding_width)), mode='constant', constant_values=padding_value)

# Convert the constant-padded array back to an image
constant_padded_image = Image.fromarray(constant_padded_array)

# Display the original and constant-padded images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(constant_padded_image, cmap='gray')
plt.title('Constant Padded Image')
plt.axis('off')

plt.tight_layout()
plt.show()


# NEAREST NEIGHBOR PADDING


# Define the padding width
padding_width = 10  # Change this value as needed

# Perform nearest neighbor padding
nearest_neighbor_padded_array = np.pad(gray_array, ((padding_width, padding_width), (padding_width, padding_width)), mode='edge')

# Convert the nearest neighbor-padded array back to an image
nearest_neighbor_padded_image = Image.fromarray(nearest_neighbor_padded_array)

# Display the original and nearest neighbor-padded images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(nearest_neighbor_padded_image, cmap='gray')
plt.title('Nearest Neighbor Padded Image')
plt.axis('off')

plt.tight_layout()
plt.show()


# REFLECT PADDING

# Define the padding width
padding_width = 10  # Change this value as needed

# Perform reflect padding
reflect_padded_array = np.pad(gray_array, ((padding_width, padding_width), (padding_width, padding_width)), mode='reflect')

# Convert the reflect-padded array back to an image
reflect_padded_image = Image.fromarray(reflect_padded_array)

# Display the original and reflect-padded images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reflect_padded_image, cmap='gray')
plt.title('Reflect Padded Image')
plt.axis('off')

plt.tight_layout()
plt.show()

'''





'''
# FILTERING

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, median_filter, maximum_filter, minimum_filter

# Load the image
image_path = 'E:/nature.jpg'  # Replace with the path to your image file
image = Image.open(image_path)

# Convert the image to grayscale
gray_image = image.convert('L')

# Convert the grayscale image to a numpy array
gray_array = np.array(gray_image)

# Define the size of the kernel (window)
kernel_size = 3  # You can adjust the kernel size as needed

# Apply mean filter (uniform filter)
mean_filtered_array = uniform_filter(gray_array, size=kernel_size)

# Apply median filter
median_filtered_array = median_filter(gray_array, size=kernel_size)

# Apply max filter
max_filtered_array = maximum_filter(gray_array, size=kernel_size)

# Apply min filter
min_filtered_array = minimum_filter(gray_array, size=kernel_size)

# Convert the filtered arrays back to images
mean_filtered_image = Image.fromarray(mean_filtered_array)
median_filtered_image = Image.fromarray(median_filtered_array)
max_filtered_image = Image.fromarray(max_filtered_array)
min_filtered_image = Image.fromarray(min_filtered_array)

# Display the original and filtered images
plt.figure(figsize=(15, 8))

# Display original image
plt.subplot(2, 3, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Display mean filtered image
plt.subplot(2, 3, 2)
plt.imshow(mean_filtered_image, cmap='gray')
plt.title('Mean Filter')
plt.axis('off')

# Display median filtered image
plt.subplot(2, 3, 3)
plt.imshow(median_filtered_image, cmap='gray')
plt.title('Median Filter')
plt.axis('off')

# Display max filtered image
plt.subplot(2, 3, 4)
plt.imshow(max_filtered_image, cmap='gray')
plt.title('Max Filter')
plt.axis('off')

# Display min filtered image
plt.subplot(2, 3, 5)
plt.imshow(min_filtered_image, cmap='gray')
plt.title('Min Filter')
plt.axis('off')

plt.tight_layout()
plt.show()

'''




'''
EDGE DETECTION


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, prewitt, laplace, gaussian_laplace
from skimage import feature

# Load the image
image_path = 'E:/nature.jpg'  # Replace with the path to your image file
image = Image.open(image_path)

# Convert the image to grayscale
gray_image = image.convert('L')

# Convert the grayscale image to a numpy array
gray_array = np.array(gray_image)

# Define the sigma for Gaussian smoothing (used in Canny and Laplacian of Gaussian)
sigma = 1.0  # You can adjust this value as needed

# Sobel filter for edge detection
sobel_x = sobel(gray_array, axis=0)  # Sobel filter in x-direction
sobel_y = sobel(gray_array, axis=1)  # Sobel filter in y-direction
sobel_edge = np.hypot(sobel_x, sobel_y)  # Magnitude of gradient

# Prewitt filter for edge detection
prewitt_x = prewitt(gray_array, axis=0)  # Prewitt filter in x-direction
prewitt_y = prewitt(gray_array, axis=1)  # Prewitt filter in y-direction
prewitt_edge = np.hypot(prewitt_x, prewitt_y)  # Magnitude of gradient

# Canny edge detection
canny_edges = feature.canny(gray_array, sigma=sigma)

# Laplacian filter for edge detection
laplacian_edges = laplace(gray_array)

# Laplacian of Gaussian filter for edge detection
log_edges = gaussian_laplace(gray_array, sigma=sigma)

# Convert the edge-detected arrays to images
sobel_image = Image.fromarray((sobel_edge * 255).astype(np.uint8))
prewitt_image = Image.fromarray((prewitt_edge * 255).astype(np.uint8))
canny_image = Image.fromarray((canny_edges * 255).astype(np.uint8))
laplacian_image = Image.fromarray((laplacian_edges * 255).astype(np.uint8))
log_image = Image.fromarray((log_edges * 255).astype(np.uint8))

# Display the original and edge-detected images
plt.figure(figsize=(15, 10))

# Display original image
plt.subplot(2, 3, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Display Sobel edge detection
plt.subplot(2, 3, 2)
plt.imshow(sobel_image, cmap='gray')
plt.title('Sobel Edge Detection')
plt.axis('off')

# Display Prewitt edge detection
plt.subplot(2, 3, 3)
plt.imshow(prewitt_image, cmap='gray')
plt.title('Prewitt Edge Detection')
plt.axis('off')

# Display Canny edge detection
plt.subplot(2, 3, 4)
plt.imshow(canny_image, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')

# Display Laplacian edge detection
plt.subplot(2, 3, 5)
plt.imshow(laplacian_image, cmap='gray')
plt.title('Laplacian Filter')
plt.axis('off')

# Display Laplacian of Gaussian edge detection
plt.subplot(2, 3, 6)
plt.imshow(log_image, cmap='gray')
plt.title('Laplacian of Gaussian')
plt.axis('off')

plt.tight_layout()
plt.show()

'''




'''

IMAGE ENHANCEMENT

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image_path = 'E:/nature.jpg'  # Replace with the path to your image file
image = Image.open(image_path)

# Convert the image to grayscale
gray_image = image.convert('L')

# Convert the grayscale image to a numpy array
gray_array = np.array(gray_image)

# Function for power law transformation (gamma correction)
def power_law_transformation(image_array, gamma):
    # Apply the transformation: output = (input / 255) ** gamma * 255
    transformed_array = ((image_array / 255) ** gamma) * 255
    transformed_array = transformed_array.astype(np.uint8)
    return transformed_array

# Function for log transformation
def log_transformation(image_array):
    # Apply the log transformation: output = c * log(1 + input)
    c = 255 / np.log(1 + np.max(image_array))  # Calculate constant c
    transformed_array = c * np.log(1 + image_array)
    transformed_array = transformed_array.astype(np.uint8)
    return transformed_array

# Function for histogram equalization
def histogram_equalization(image_array):
    # Calculate the histogram of the image
    histogram, bins = np.histogram(image_array, bins=256, range=(0, 256))
    
    # Calculate the cumulative distribution function (CDF)
    cdf = np.cumsum(histogram)
    cdf = cdf / cdf[-1]  # Normalize the CDF
    
    # Create a mapping from original gray levels to equalized gray levels
    equalized_mapping = np.round(cdf * 255).astype(np.uint8)
    
    # Apply the mapping to the original image array
    equalized_array = equalized_mapping[image_array]
    
    return equalized_array

# Apply power law transformation (gamma correction)
gamma = 2.2  # Adjust the gamma value as needed
power_law_array = power_law_transformation(gray_array, gamma)

# Apply log transformation
log_array = log_transformation(gray_array)

# Apply histogram equalization
hist_eq_array = histogram_equalization(gray_array)

# Convert the transformed arrays back to images
power_law_image = Image.fromarray(power_law_array)
log_image = Image.fromarray(log_array)
hist_eq_image = Image.fromarray(hist_eq_array)

# Display the original and transformed images
plt.figure(figsize=(15, 6))

# Display original image
plt.subplot(1, 4, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Display power law (gamma) transformation
plt.subplot(1, 4, 2)
plt.imshow(power_law_image, cmap='gray')
plt.title(f'Power Law (Gamma = {gamma})')
plt.axis('off')

# Display log transformation
plt.subplot(1, 4, 3)
plt.imshow(log_image, cmap='gray')
plt.title('Log Transformation')
plt.axis('off')

# Display histogram equalization
plt.subplot(1, 4, 4)
plt.imshow(hist_eq_image, cmap='gray')
plt.title('Histogram Equalization')
plt.axis('off')

plt.tight_layout()
plt.show()

'''



'''
# MORPHOLOGICAL OPERATIONS

NOT DONE!!!!!!!!!!
'''



'''
FOURIER  FFT

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image and convert it to grayscale
image_path = 'E:/nature.jpg'  # Replace with the path to your image file
image = Image.open(image_path)
gray_image = image.convert('L')

# Convert the grayscale image to a numpy array
gray_array = np.array(gray_image)

# Apply Fast Fourier Transform
fft_result = np.fft.fft2(gray_array)

# Shift the zero frequency component to the center of the spectrum
fft_shifted = np.fft.fftshift(fft_result)

# Calculate the magnitude spectrum
magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)

# Display the magnitude spectrum
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('FFT Magnitude Spectrum')
plt.axis('off')
plt.show()

'''


'''
FOURIER DCT

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import dct

# Load the image and convert it to grayscale
image_path = 'E:/nature.jpg'  # Replace with the path to your image file
image = Image.open(image_path)
gray_image = image.convert('L')

# Convert the grayscale image to a numpy array
gray_array = np.array(gray_image)

# Apply Discrete Cosine Transform
dct_result = dct(dct(gray_array.T, norm='ortho').T, norm='ortho')

# Display the DCT result
plt.imshow(np.abs(dct_result), cmap='gray')
plt.title('DCT Result')
plt.axis('off')
plt.show()

'''


'''
CONVOLUTION

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# Load the image and convert it to grayscale
image_path = 'E:/nature.jpg'  # Replace with the path to your image file
image = Image.open(image_path)
gray_image = image.convert('L')

# Convert the grayscale image to a numpy array
gray_array = np.array(gray_image)

# Define a filter kernel (e.g., a simple blur filter)
filter_kernel = np.array([[1/9, 1/9, 1/9],
                          [1/9, 1/9, 1/9],
                          [1/9, 1/9, 1/9]])

# Apply convolution using the filter kernel
convolved_array = convolve(gray_array, filter_kernel)

# Display the convolved image
plt.imshow(convolved_array, cmap='gray')
plt.title('Convolved Image')
plt.axis('off')
plt.show()

'''


'''
FOURIER FILTERS

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage

# Load the image and convert to grayscale
image_path = 'E:/nature.jpg'  # Replace with the path to your image file
image = Image.open(image_path)
gray_image = image.convert('L')

# Convert the grayscale image to a numpy array
gray_array = np.array(gray_image)

# Define a function to create a grid of frequencies
def create_frequency_grid(shape):
    rows, cols = shape
    x = np.fft.fftfreq(cols)
    y = np.fft.fftfreq(rows)
    X, Y = np.meshgrid(x, y)
    return np.sqrt(X ** 2 + Y ** 2)

# Define the different filters

# Ideal Low-Pass Filter
def ideal_low_pass_filter(shape, cutoff):
    grid = create_frequency_grid(shape)
    return (grid <= cutoff).astype(np.float32)

# Butterworth Low-Pass Filter
def butterworth_low_pass_filter(shape, cutoff, order):
    grid = create_frequency_grid(shape)
    return 1 / (1 + (grid / cutoff) ** (2 * order))

# Gaussian Low-Pass Filter
def gaussian_low_pass_filter(shape, cutoff):
    grid = create_frequency_grid(shape)
    return np.exp(-(grid ** 2) / (2 * cutoff ** 2))

# Ideal High-Pass Filter
def ideal_high_pass_filter(shape, cutoff):
    return 1 - ideal_low_pass_filter(shape, cutoff)

# Butterworth High-Pass Filter
def butterworth_high_pass_filter(shape, cutoff, order):
    return 1 - butterworth_low_pass_filter(shape, cutoff, order)

# Gaussian High-Pass Filter
def gaussian_high_pass_filter(shape, cutoff):
    return 1 - gaussian_low_pass_filter(shape, cutoff)

# Band-Pass Filter
def band_pass_filter(shape, low_cutoff, high_cutoff, order):
    return butterworth_low_pass_filter(shape, high_cutoff, order) - butterworth_high_pass_filter(shape, low_cutoff, order)

# Apply a given filter to the image in the frequency domain
def apply_filter(image_array, filter_function, *args):
    # Compute the FFT of the image
    fft_image = np.fft.fft2(image_array)
    fft_image_shifted = np.fft.fftshift(fft_image)
    
    # Create the filter
    filter_mask = filter_function(image_array.shape, *args)
    
    # Apply the filter
    filtered_fft = fft_image_shifted * filter_mask
    
    # Shift back and compute the inverse FFT
    fft_image_back = np.fft.ifftshift(filtered_fft)
    filtered_image = np.fft.ifft2(fft_image_back)
    
    return np.abs(filtered_image)

# Define cutoff frequencies and filter order
low_cutoff = 0.1  # Cutoff frequency for low-pass filters
high_cutoff = 0.2  # Cutoff frequency for high-pass filters
order = 2  # Order of the Butterworth filters

# Apply each filter and display the result

# Ideal Low-Pass Filter
ideal_lp_filtered = apply_filter(gray_array, ideal_low_pass_filter, low_cutoff)
plt.imshow(ideal_lp_filtered, cmap='gray')
plt.title('Ideal Low-Pass Filter')
plt.axis('off')
plt.show()

# Butterworth Low-Pass Filter
butterworth_lp_filtered = apply_filter(gray_array, butterworth_low_pass_filter, low_cutoff, order)
plt.imshow(butterworth_lp_filtered, cmap='gray')
plt.title('Butterworth Low-Pass Filter')
plt.axis('off')
plt.show()

# Gaussian Low-Pass Filter
gaussian_lp_filtered = apply_filter(gray_array, gaussian_low_pass_filter, low_cutoff)
plt.imshow(gaussian_lp_filtered, cmap='gray')
plt.title('Gaussian Low-Pass Filter')
plt.axis('off')
plt.show()

# Ideal High-Pass Filter
ideal_hp_filtered = apply_filter(gray_array, ideal_high_pass_filter, high_cutoff)
plt.imshow(ideal_hp_filtered, cmap='gray')
plt.title('Ideal High-Pass Filter')
plt.axis('off')
plt.show()

# Butterworth High-Pass Filter
butterworth_hp_filtered = apply_filter(gray_array, butterworth_high_pass_filter, high_cutoff, order)
plt.imshow(butterworth_hp_filtered, cmap='gray')
plt.title('Butterworth High-Pass Filter')
plt.axis('off')
plt.show()

# Gaussian High-Pass Filter
gaussian_hp_filtered = apply_filter(gray_array, gaussian_high_pass_filter, high_cutoff)
plt.imshow(gaussian_hp_filtered, cmap='gray')
plt.title('Gaussian High-Pass Filter')
plt.axis('off')
plt.show()

# Band-Pass Filter
band_pass_filtered = apply_filter(gray_array, band_pass_filter, low_cutoff, high_cutoff, order)
plt.imshow(band_pass_filtered, cmap='gray')
plt.title('Band-Pass Filter')
plt.axis('off')
plt.show()

'''




'''
HISTOGRAM SEGMENTATION

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image and convert to grayscale
image_path = 'E:/nature.jpg'  # Replace with the path to your image file
image = Image.open(image_path)
gray_image = image.convert('L')

# Convert the grayscale image to a numpy array
gray_array = np.array(gray_image)

# Compute the histogram of the grayscale image
histogram, bins = np.histogram(gray_array, bins=256, range=(0, 256))

# Plot the histogram
plt.figure()
plt.bar(bins[:-1], histogram, width=1, edgecolor='black')
plt.title('Histogram of Grayscale Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

# Choose thresholds based on the histogram
# For simplicity, let's assume the image has two peaks, and we choose the valley between them as the threshold
# In practice, you may use automated methods such as Otsu's method to choose the threshold

# Example threshold value (you may adjust this value based on the histogram)
threshold = 128

# Segment the image using the threshold
segmented_array = (gray_array > threshold).astype(np.uint8) * 255

# Display the original and segmented images
plt.figure(figsize=(12, 6))

# Original grayscale image
plt.subplot(1, 2, 1)
plt.imshow(gray_array, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# Segmented image
plt.subplot(1, 2, 2)
plt.imshow(segmented_array, cmap='gray')
plt.title('Segmented Image')
plt.axis('off')

plt.tight_layout()
plt.show()

'''



'''
CANNY EDGE SEGMENTATION

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import feature

# Load the image and convert to grayscale
image_path = 'E:/nature.jpg'  # Replace with the path to your image file
image = Image.open(image_path)
gray_image = image.convert('L')

# Convert the grayscale image to a numpy array
gray_array = np.array(gray_image)

# Perform Canny edge detection
edges = feature.canny(gray_array)

# Display the original image and the edge-detected image
plt.figure(figsize=(12, 6))

# Original grayscale image
plt.subplot(1, 2, 1)
plt.imshow(gray_array, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# Edge-detected image
plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edge Segmentation (Canny)')
plt.axis('off')

plt.tight_layout()
plt.show()

'''



'''
DIFFERENTIAL EQUATION BASED SEGMENTATION

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import restoration

# Load the image and convert to grayscale
image_path = 'E:/nature.jpg'  # Replace with the path to your image file
image = Image.open(image_path)
gray_image = image.convert('L')

# Convert the grayscale image to a numpy array
gray_array = np.array(gray_image)

# Perform anisotropic diffusion
diffused_array = restoration.denoise_tv_chambolle(gray_array, weight=0.1)

# Display the original image and the diffused image
plt.figure(figsize=(12, 6))

# Original grayscale image
plt.subplot(1, 2, 1)
plt.imshow(gray_array, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# Diffused image
plt.subplot(1, 2, 2)
plt.imshow(diffused_array, cmap='gray')
plt.title('Differential Equation-Based Segmentation (Anisotropic Diffusion)')
plt.axis('off')

plt.tight_layout()
plt.show()

'''



'''
MODEL BASED SEGMENTATION

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import segmentation
from skimage.color import rgb2gray

# Load the image
image_path = 'E:/nature.jpg'  # Replace with the path to your image file
image = Image.open(image_path)

# Convert the image to grayscale
gray_image = image.convert('L')

# Convert the grayscale image to a numpy array
gray_array = np.array(gray_image)

# Initialize the snake (active contour) as a circle in the middle of the image
s = np.linspace(0, 2 * np.pi, 100)
init = 150 + 100 * np.array([np.cos(s), np.sin(s)]).T

# Perform active contour segmentation (snake)
snake = segmentation.active_contour(gray_array, init, alpha=0.01, beta=0.1, gamma=0.001)

# Plot the original image and the segmented image
plt.figure(figsize=(12, 6))

# Original grayscale image
plt.subplot(1, 2, 1)
plt.imshow(gray_array, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# Segmented image with snake (active contour)
plt.subplot(1, 2, 2)
plt.imshow(gray_array, cmap='gray')
plt.plot(snake[:, 1], snake[:, 0], '-r', linewidth=2)
plt.title('Model-Based Segmentation (Active Contour)')
plt.axis('off')

plt.tight_layout()
plt.show()

'''


'''
NOISE REMOVAL

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

# Load the image
image_path = 'E:/nature.jpg'  # Replace with the path to your image file
image = Image.open(image_path)

# Convert the image to grayscale for easier processing
gray_image = image.convert('L')

# Apply Gaussian Blur for noise removal
gaussian_blur_image = gray_image.filter(ImageFilter.GaussianBlur(radius=2))

# Apply Median Filter for noise removal
median_filter_image = gray_image.filter(ImageFilter.MedianFilter(size=3))

# Display the original image, original grayscale image, Gaussian blur, and median filter images
plt.figure(figsize=(16, 4))

# Original color image
plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title('Original Color Image')
plt.axis('off')

# Original grayscale image
plt.subplot(2, 2, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# Gaussian blur filtered image
plt.subplot(2, 2, 3)
plt.imshow(gaussian_blur_image, cmap='gray')
plt.title('Gaussian Blur Filter')
plt.axis('off')

# Median filter image
plt.subplot(2, 2, 4)
plt.imshow(median_filter_image, cmap='gray')
plt.title('Median Filter')
plt.axis('off')

# Adjust the layout for better spacing
plt.tight_layout()
# Show the plots
plt.show()


'''


'''
SAMPLING AND QUANTIZATION


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image and convert to grayscale
image_path = 'E:/nature.jpg'  # Replace with the path to your image file
image = Image.open(image_path)
gray_image = image.convert('L')

# Convert the grayscale image to a numpy array
gray_array = np.array(gray_image)

# Downsampling: Reduce the image resolution by taking every nth pixel
n = 2  # Downsampling factor
downsampled_array = gray_array[::n, ::n]

# Upsampling: Increase the image resolution using interpolation
# We'll use PIL's resize method with bilinear interpolation
upsampled_array = Image.fromarray(downsampled_array).resize(gray_array.shape[::-1], Image.BILINEAR)
upsampled_array = np.array(upsampled_array)

# Quantization: Reduce the number of gray levels
num_levels = 16  # Number of gray levels (e.g., 4, 8, 16, 32, etc.)
quantized_array = np.round((gray_array / 255) * (num_levels - 1))
quantized_array = (quantized_array / (num_levels - 1)) * 255
quantized_array = quantized_array.astype(np.uint8)

# Display the original image, downsampled, upsampled, and quantized images
plt.figure(figsize=(12, 12))

# Original color image
plt.subplot(3, 2, 1)
plt.imshow(image)
plt.title('Original Color Image')
plt.axis('off')

# Original grayscale image
plt.subplot(3, 2, 2)
plt.imshow(gray_array, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# Downsampled image
plt.subplot(3, 2, 3)
plt.imshow(downsampled_array, cmap='gray')
plt.title(f'Downsampled Image (factor: {n})')
plt.axis('off')

# Upsampled image
plt.subplot(3, 2, 4)
plt.imshow(upsampled_array, cmap='gray')
plt.title('Upsampled Image')
plt.axis('off')

# Quantized image
plt.subplot(3, 2, 5)
plt.imshow(quantized_array, cmap='gray')
plt.title(f'Quantized Image ({num_levels} gray levels)')
plt.axis('off')

# Add one more subplot here if you want to display another image.
# For example, you could display another version of quantization or another process

plt.tight_layout()
plt.show()

'''





