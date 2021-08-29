import numpy as np
import matplotlib.pyplot as plt


from astropy.convolution import RickerWavelet2DKernel
ricker_2d_kernel = RickerWavelet2DKernel(5)
plt.imshow(ricker_2d_kernel, interpolation='none', origin='lower')
plt.xlabel('x [pixels]')
plt.ylabel('y [pixels]')
plt.colorbar()
plt.show()

print(ricker_2d_kernel)
