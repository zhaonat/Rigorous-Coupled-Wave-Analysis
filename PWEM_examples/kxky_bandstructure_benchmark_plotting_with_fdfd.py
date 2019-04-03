import sys
import os
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from mpl_toolkits.mplot3d import Axes3D


matlab_data =os.path.join('kxky_photonic_circle_bandstructure.mat');
mat = scipy.io.loadmat(matlab_data)
print(mat.keys())
wvlen_scan = np.squeeze(mat['wvlen_scan']);
omega_scan = 1/wvlen_scan;
ky_spectra = np.squeeze(mat['ky_spectra']);
print(ky_spectra.shape)
ky_scan = np.linspace(-np.pi, np.pi, 400);

X,Y = np.meshgrid(omega_scan, ky_scan);
print(X.shape)

#first dimension is ky... second dimension is kx...

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X, Y, np.real(ky_spectra[:,:,0]), marker='.')

plt.show();