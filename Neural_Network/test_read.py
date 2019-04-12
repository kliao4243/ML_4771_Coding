import os
# fetching a random png image from my home directory, which has size 258 x 384
img_file = os.path.expanduser("test_1.png")

from scipy import misc
# read this image in as a NumPy array, using imread from scipy.misc
M = misc.imread(img_file)

M.shape