import numpy as np

from skimage import io
from skimage.transform import resize, rotate
from skimage.draw import polygon, ellipse

from numpy.fft import ifftshift, ifft2, fftshift, fft2
from numpy import exp, pi, mean, var, std, abs, sin, cos, sum

data = np.load('/home/jamunoz/.keras/datasets/mnist.npz')

def generate_ellipse():
	x = np.random.randint(high=320, low=192)
	y = np.random.randint(high=320, low=192)
	scale = abs(np.random.normal(scale=0.5))
	r_radius = 100 * scale
	c_radius = 200 * scale
	rot_angle = np.random.randint(low=0, high=359)

	img = np.zeros((512, 512), dtype=np.double)
	rr, cc = ellipse(x, y, r_radius, c_radius, img.shape)
	img[rr, cc] = 1
	
	img = rotate(img, rot_angle)
	
	if sum(sum(img)) < 10000:
		img = generate_ellipse()
		
	return img

img = generate_ellipse()



#print(len(data))
#print(type(data))

#lst = data.files

#for item in lst:
#	print(item)
#	print(data[item])
#
#	print(len(data[item]))
#	print(type(data[item]))
#
#print(data['x_train'][0].shape)
#
#print('yes')
