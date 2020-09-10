from scipy import misc
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np

face = misc.face(gray = True)

blurred_face = ndimage.gaussian_filter(face, sigma=5)
difference = np.subtract(blurred_face, face)

plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.imshow(face, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(132)
plt.imshow(blurred_face, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(133)
plt.imshow(difference, cmap=plt.cm.gray)
plt.axis('off')

plt.subplots_adjust(wspace=0, hspace=0., top=0.99, bottom=0.01,
                    left=0.01, right=0.99)

plt.show()
