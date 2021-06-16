# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 12:00:47 2021

@author: Asus
"""
import time
initial_start_time = time.time() # here we store the time at which the program starts.

### PART 2: IMAGE PROCESSING

# We will start by importing the requested libraries
import matplotlib.pyplot as plt # to show and save the images
import matplotlib.gridspec as gridspec # to show the images.
import imageio # to read the images from the data file
import skimage.color as color # to convert images from one type to another
import skimage.feature as feature
import skimage.transform as transform
import skimage.filters as filters # for the optimal threshold value, or the Gaussian Mask.
import skimage.util as util # to add random noise to an image.
from scipy import ndimage # to apply a Box or Average Smoothing Mask.
from sklearn.cluster import KMeans # to apply image segmentation.
from skimage.segmentation import slic # to apply image segmentation.



# 1. [8 points] Determine the size of the avengers imdb.jpg image. Produce a grayscale and a black-
# and-white representation of it.


print("\nQUESTION 1: Avengers movie cover\n")

# We will start by reading the image with the help of imageio.
avengers = imageio.imread('data/image_data/avengers_imdb.jpg', pilmode='RGB')

# First, we will show the image so anyone executing the program knows what we are talking about.
print("This question is based on the following image:")
plt.imshow(avengers) # we create the image.
plt.axis('off') # we remove the axis.
plt.show() # we show it.

# Now, we will print the shape (size) of this image
print("\nThe size of the Avengers movie cover is {}".format(avengers.shape))

# Next step is to convert the RGB image into a greyscale format. We can help ourselves with the 
# scikit-image library. It has a package called color that does this perfectly.
avengers_grayscale = color.rgb2gray(avengers) # we convert the image to a grayscale representation

# We have to save this image to the 'outputs' file.
plt.imsave('outputs/avengers_grayscale.jpg', avengers_grayscale, cmap=plt.cm.gray) # we save the image in the 'outputs' folder

# Now we have to convert it into a binary representation. This kind of representations have their
# pixel values set to 0 (black) or 1 (white). We will use imageio.filter.threshold_otsu this time.
# This built-in function returns the optimal threshold by applying Otsu's method.
threshold = filters.threshold_otsu(avengers_grayscale) # to estimate the optimal threshold

# Now we can convert the image into black-and-white.
avengers_binary = avengers_grayscale > threshold

# Like before, we save the image.
plt.imsave('outputs/avengers_binary.jpg', avengers_binary, cmap=plt.cm.gray) # we save the image in the 'outputs' folder

# Lastly, we will plot the three images together so we can see the conversion from one to
# another.
print("\nThe original image and its respective representations are the following:")

# We create the figure and the subplots (1 row and 3 columns).
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(16, 6), sharex=True, sharey=True )

ax0.imshow(avengers) # original Avengers movie cover.
ax0.axis('off') # remove axis.
ax0.set_title('Original Image', fontweight='bold') # set title.

ax1.imshow(avengers_grayscale, cmap=plt.cm.gray, interpolation='nearest') # grayscale representation.
ax1.axis('off') # remove axis.
ax1.set_title('Grayscale Representation', fontweight='bold') # set titlte.

ax2.imshow(avengers_binary, cmap=plt.cm.gray, interpolation='nearest') # B&W representation.
ax2.axis('off') # remove axis.
ax2.set_title('Binary Representation', fontweight='bold') # set title.

fig.tight_layout()
plt.show() # render plot.



# 2. [12 points] Add Gaussian random noise in bush_house_wikipedia.jpg (with variance 0.1) and 
# filter the perturbed image with a Gaussian mask (sigma equal to 1) and a uniform smoothing mask
# (the latter of size 9x9).


print("\n\nQUESTION 2: Bush House (Wikipedia)\n")

# We will start by reading the image with the help of imageio.
bush_house = imageio.imread('data/image_data/bush_house_wikipedia.jpg')

# First, we will show the image so anyone executing the program knows what we are talking about.
print("This question is based on the following image:")
plt.imshow(bush_house) # we create the image.
plt.axis('off') # we remove the axis.
plt.show() # we show it.

# The first step of this question is to add Gaussian noise to the image. We will do so by applying
# the scikit-image built-in function random_noise. We can apply Gaussian noise with a specific 
# variance, you have to pass the variance you want to the parameter 'var'.
perturbed_bush_house = util.random_noise(bush_house, mode='gaussian', var=0.1)

# We save the perturbed image in our 'outputs' folder:
plt.imsave('outputs/perturbed_bush_house.jpg', perturbed_bush_house) # we save the image in the 'outputs' folder.

# Now that we have the perturbed image, we can apply the different smoothing techniques to 
# compare them. We will start by applying the Gaussian Mask. This can be easily done with the
# built-in function of skimage: filters.gaussian. We set the 'multichannel' parameter to True 
# so each channel (Red, Green and Blue) is filtered separately.
gaussian_bush_house = filters.gaussian(perturbed_bush_house, sigma=1, multichannel=True)

# Now we will save the image in the 'outputs' folder.
plt.imsave('outputs/gaussian_bush_house.jpg', gaussian_bush_house)

# The next mask we have to apply is the Box Mask or Average Smoothing Mask, where each pixel 
# is replaced by the average of itself and their neighbours. This can be performed with SciPy, 
# which has a built-in function called uniform_filter that performs a multidimensional uniform 
# filter. This function has a parameter called 'size' which specifies the sizes of the uniform 
# filter. As it is a colored image, we have to pass a 2D matrix so that the function does not
# average all three colour channels.
uniform_bush_house = ndimage.uniform_filter(perturbed_bush_house, size=(9,9,1))

# Now we will save the image in the 'outputs' folder.
plt.imsave('outputs/uniform_bush_house.jpg', uniform_bush_house)

# Lastly, we will plot the four images together so we can see the conversion from one to
# another.
print("\nThe original image and its respective alterations are the following:")

# First, we create the figure and each subplot:
fig, ([ax0, ax1], [ax2, ax3]) = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))

ax0.imshow(bush_house) # original Bush House image.
ax0.axis('off') # remove axis.
ax0.set_title('Original Image', fontweight='bold') # set title.

ax1.imshow(perturbed_bush_house, interpolation='nearest') # with Gaussian noise.
ax1.axis('off') # remove axis.
ax1.set_title('Image with Gaussian noise', fontweight='bold') # set titlte.

ax2.imshow(gaussian_bush_house, interpolation='nearest') # with Gaussian Mask.
ax2.axis('off') # remove axis.
ax2.set_title('Gaussian Mask', fontweight='bold') # set title.

ax3.imshow(uniform_bush_house, interpolation='nearest') # with Uniform Mask.
ax3.axis('off') # remove axis.
ax3.set_title('Uniform Mask', fontweight='bold') # set title.

fig.tight_layout()
plt.show() # render plot.



# 3. [8 points] Divide forestry_commission_gov_uk.jpg into 5 segments using k-means segmentation.


print("\n\nQUESTION 3: Landscape (Forestry Commission)\n")

# We will start by reading the image with the help of imageio.
landscape = imageio.imread('data/image_data/forestry_commission_gov_uk.jpg')

# First, we will show the image so anyone executing the program knows what we are talking about.
print("This question is based on the following image:")
plt.imshow(landscape) # we create the image.
plt.axis('off') # we remove the axis.
plt.show() # we show it.

# Now we will perform a k-means clustering to this image. The clustering can be made using
# coulour alone, or using colour and position. We will perform both of them. As the question
# description indicates us, we have to select 5 different sections, so we set 'k' to 5. 
n_segments = 5 # we set the number of clusters we want.

# First we will perform a k-means clustering only with the coulour. For this, we need a dataset
# that has three columns and n rows, where n is the number of pixels. Each column will be the
# intensity of red, green and blue, as it is a RGB digital image. This can be easily obtained with 
# the .reshapee() NumPy function:
colours = landscape.reshape((landscape.shape[0]*landscape.shape[1], landscape.shape[2]))
            
# Now that we have our 'colours' array created, we can perform the k-means algorithm on it.
kmeans_colour = KMeans(n_clusters=n_segments, random_state=420) # we initialise the algorithm.
kmeans_colour.fit(colours) # we train the model on our data.

# Once the model is trained, we will store the labels in 'colour_labels'. After that, we will
# have to create a new image from the labels, giving each pixel a label. So, we have to convert
# the 1D label array to a 2D matrix.
colour_labels = kmeans_colour.labels_ # we store the labels in this variable.

# Once we have the pixel labels, we can reshape the 'colour_labels' array to a image-shaped array like 
# the one we had at the beginning. As we only have a label per pixel, we reshape it to a 2D array like
# one of the colour intensities.
colour_segmentation = colour_labels.reshape(landscape.shape[0], landscape.shape[1])

# Now, we will save this image in the 'outputs' folder:
plt.imsave('outputs/colour_segmentation.jpg', colour_segmentation)

# Now, we will perform the same type of clustering, but this time we will add two columns more,
# one for the i position, and another one for the j position. This way, the algorithm also takes
# into consideration the location of the pixel. This part is easier to program as there is a skimage
# package that does it directly, 'segmentation'. We are going to use the 'slic' function.

# We can perform the k-means algorithm directly on the image. We have to choose a number of parameters:
    # image: input image.
    # n_segments: approximate number of labels in the segmented output image.
    # compactness: balances colour proximity and space proximity.
    # sigma: width of Gaussian smoothing kernel for pre-processing for each dimension of the image.
    # multichannel: whether the last axis of the image is to be interpreted as multiple channels or another spatial dimension.
col_pos_segmentation = slic(image=landscape, n_segments=n_segments, compactness=20, sigma=0.5, multichannel=True, start_label=1)

# Now, we will save this image in the 'outputs' folder:
plt.imsave('outputs/colour_position_segmentation.jpg', col_pos_segmentation)

# In order to interpret better these results, we will take hand of a function called 'label2rgb'. This 
# function basically superimposes the original image and the segmented one. It has a parameter 'alpha'
# which sets the opacity of colorized labels. We have set this parameter to 0.5 so it can be interpreted.

# First, we do it with the colour segmentation only:
superimposed_colour = color.label2rgb(label=colour_segmentation, image=landscape, alpha=0.5, bg_label=5)

# Now, we will save this image in the 'outputs' folder:
plt.imsave('outputs/colour_superimposed_segmentation.jpg', superimposed_colour)

# Now, we also do it with the colour and position segmentation:
superimposed_colour_position = color.label2rgb(label=col_pos_segmentation, image=landscape, alpha=0.5, bg_label=5)

# Now, we will save this image in the 'outputs' folder:
plt.imsave('outputs/colour_position_superimposed_segmentation.jpg', superimposed_colour_position)

# Lastly, we will plot the five images together so we can see the segmentation from one to
# another.
print("\nThe original image and its respective segmentations are the following:")

# First, we create the figure and each subplot:
fig = plt.figure(figsize=(16, 16), tight_layout=True)
gs = gridspec.GridSpec(3, 2)

ax1 = fig.add_subplot(gs[0, :]) # add the subplot.
ax1.imshow(landscape) # original image.
ax1.axis('off') # remove axis.
ax1.set_title('Original Image', fontweight='bold') # set title.

ax2 = fig.add_subplot(gs[1, 0]) # add the subplot.
ax2.imshow(colour_segmentation) # with the colour segmentation.
ax2.axis('off') # remove axis.
ax2.set_title('Colour Segmentation', fontweight='bold') # set titlte.

ax3 = fig.add_subplot(gs[1, 1]) # add the subplot.
ax3.imshow(superimposed_colour, interpolation='nearest') # with the superimposed original image.
ax3.axis('off') # remove axis.
ax3.set_title('Superimposed Colour Segmentation', fontweight='bold') # set title.

ax4 = fig.add_subplot(gs[2, 0]) # add the subplot.
ax4.imshow(col_pos_segmentation) # with the colour-position segmentation.
ax4.axis('off') # remove axis.
ax4.set_title('Colour&Position Segmentation', fontweight='bold') # set titlte.

ax5 = fig.add_subplot(gs[2, 1]) # add the subplot.
ax5.imshow(superimposed_colour_position, interpolation='nearest') # with the superimposed original image.
ax5.axis('off') # remove axis.
ax5.set_title('Superimposed Colour&Position Segmentation', fontweight='bold') # set title.

fig.tight_layout()
plt.show() # render plot.



# 4. [12 points] Perform Canny edge detection and apply Hough transform on rolland_garros_tv5monde.jpg.


print("\n\nQUESTION 4: Rolland Garros Image (TV5Monde)\n")

# We will start by reading the image with the help of imageio.
rg = imageio.imread('data/image_data/rolland_garros_tv5monde.jpg')

# First, we will show the image so anyone executing the program knows what we are talking about.
print("This question is based on the following image:")
plt.imshow(rg) # we create the image.
plt.axis('off') # we remove the axis.
plt.show() # we show it.

# Now we will perform two edge detection algorithms. We will start by the Canny Edge Detection one. This
# algorithm can be performed with a built-in function of scikit image. The package is called 'feature' and
# the function is called 'canny'.

# We first have to convert the original image to greyscale. We do so by applying the 'rgb2gray' function.
gray_rg = color.rgb2gray(rg)

# Now we are in position of performing Canny edge detection. We do so with the built-in function we 
# mentioned before. 
canny_rg = feature.canny(gray_rg)

# Now, we will save this image in the 'outputs' folder:
plt.imsave('outputs/hough_transform_rolland_garros.jpg', canny_rg, cmap=plt.cm.gray)

# Now we will apply the Hough Transform algorithm helping ourselves with the built-in function of
# skimage.transform called 'probabilistic_hough_line'.  We have to choose a number of parameters:
    # image: input image.
    # threshold: the threshold, the default value is 10.
    # line_length: Minimum accepted length of detected lines. Increase the parameter to extract longer lines.
                 # Default value is 50 and we will set it to 45 to obtain all the lines of the court.
    # line_gap: Maximum gap between pixels to still form a line. Increase the parameter to merge broken lines 
              # more aggressively. Default value is 10, but we will reduce it to 5 so it does not add the tennis
              # players or too many lines in the stand.
              
# We can directly apply the classic straight-line Hough transform
hough_rg = transform.probabilistic_hough_line(image=canny_rg, threshold=10, line_length=45, line_gap=5)

# Now we plot the image as we want. Some preprocessing is needed as the algorithm does not return an image.
# Instead, it returns lines from a progressive probabilistic line Hough Transform.
for line in hough_rg: # loop through all the lines.
    p0, p1 = line # extract the probabilities,
    plt.plot((p0[0], p1[0]), (p0[1], p1[1])) # and plot them.
plt.ylim((gray_rg.shape[0], 0)) # we set the limits in the y-axis so the image is not inverted.
plt.axis('off') # we remove the axis.
plt.savefig('outputs/canny_edge_rolland_garros.jpg') # we save it.

# Finally, we will plot the three images to show the transformation from the original to the other two:
print("\nThe original image and its respective transformations are the following:")

# We create the figure and the subplots (1 row and 3 columns).
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(16, 6), sharex=True, sharey=True )

ax0.imshow(rg) # original Rolland Garros court.
ax0.axis('off') # remove axis.
ax0.set_title('Original Tennis Court', fontweight='bold') # set title.

ax1.imshow(canny_rg, cmap=plt.cm.gray, interpolation='nearest') # Canny Transformation.
ax1.axis('off') # remove axis.
ax1.set_title('Canny Edge Detection', fontweight='bold') # set titlte.

for line in hough_rg: # loop through all the lines.
    p0, p1 = line # extract the probabilities,
    ax2.plot((p0[0], p1[0]), (p0[1], p1[1])) # and plot the Hough Transform.
ax2.set_ylim(( gray_rg.shape[0], 0 )) # we set the limits in the y-axis so the image is not inverted.
ax2.axis('off') # remove axis.
ax2.set_title('Hough Transform', fontweight='bold') # set title.

fig.tight_layout()
plt.show() # render plot.

print("\nTOTAL EXECUTION TIME: {:.3f} seconds".format(time.time()-initial_start_time))