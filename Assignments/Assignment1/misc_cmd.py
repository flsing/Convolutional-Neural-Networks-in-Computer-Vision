# see http://www.scipy-lectures.org/advanced/image_processing/

from pathlib import Path

fname = 'faces/an2i/an2i_left_neutral_open.pgm'
an2i = Path( fname )
an2i.exists()

import PIL.Image
an2i_im = PIL.Image.open(fname)

# or we can use the misc package that in turn uses PIL
from scipy import misc
an2i_im = misc.imread(fname)

import importlib
# importlib.reload(pgm2pil)


# not needed anymore changed pgm2pil
# import numpy as np
# an2i_im_up = np.flipud(an2i_im)

import matplotlib.pyplot as plt

# make it non-blocking -- only needed once
plt.ion()

plt.imshow(an2i_im, cmap=plt.cm.gray)
# Only needed if blocking
# plt.show()

