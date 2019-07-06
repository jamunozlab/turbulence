import sys
import numpy as np

from skimage import io
from skimage.transform import resize, rotate
from skimage.draw import polygon, ellipse

from numpy.fft import ifftshift, ifft2, fftshift, fft2
from numpy import exp, pi, mean, var, std, abs, sin, cos, sum

sys.path.append('../')
from turbulence.utils.ft import ft_sh_phase_screen

def generate_ellipse(N=256):
	x = np.random.randint(high=int(0.90*N), low=int(0.10*N))
	y = np.random.randint(high=int(0.90*N), low=int(0.10*N))
	scale = abs(np.random.normal(scale=0.2))
	r_radius = 100 * scale
	c_radius = 200 * scale
	rot_angle = np.random.randint(low=0, high=359)

	img = np.zeros((N, N), dtype=np.double)
	rr, cc = ellipse(x, y, r_radius, c_radius, img.shape)
	img[rr, cc] = 1
	
	img = rotate(img, rot_angle)
	
	if sum(sum(img)) < 10000: # Make sure that fig is not too tiny
		img = generate_ellipse()
		
	return img

def generate_poly():

    def generate_sides():
        x2 = np.random.randint(low=97, high=416)
        x1 = np.random.randint(low=96, high=x2+1)
        return x1, x2

    poly = np.array((
		  (generate_sides()),
        (generate_sides()),
        (generate_sides()),
        (generate_sides()),
    ))
										      
    img = np.zeros((512, 512), dtype=np.double)
    rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)
    img[rr, cc] = 1

    if sum(sum(img)) < 10000:
        img = generate_poly()
		  
    return img

