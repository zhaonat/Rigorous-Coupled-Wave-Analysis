{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Original paper by Kronig, Penney:\n",
    "\n",
    "R. de L. Kronig, W.G.Penney, Proc. Roy. Soc. (London) A130, 499 (1931)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 333,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import scipy.linalg as la\n",
    "import matplotlib.pylab as plt\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Solve an entire system of multiple wells"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 332,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "n_potentials = 20\n",
    "dx = 0.1\n",
    "steps_per_well = np.int(1. / dx)\n",
    "potential_width = np.int(0.4 / dx)\n",
    "n_x = np.int(n_potentials / dx)\n",
    "V_0 = -100\n",
    "\n",
    "V = np.zeros(n_x)\n",
    "\n",
    "for i in range(n_potentials):\n",
    "    left = i * steps_per_well\n",
    "    right = i * steps_per_well + potential_width\n",
    "    V[left:right] = V_0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 334,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "H = np.zeros((n_x, n_x))\n",
    "np.fill_diagonal(H, 2 * 1 + dx * dx * V)\n",
    "H = H + np.diag(-1 * np.ones(n_x - 1), -1) \n",
    "H = H + np.diag(-1 * np.ones(n_x - 1), +1) \n",
    "\n",
    "H[0, -1] = -1\n",
    "H[-1, 0] = -1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 335,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "evals, evecs = la.eigh(H)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 336,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "plt.plot(evecs[:, 4], color=\"red\", linewidth=3, zorder=10)\n",
    "ax_t = plt.twinx()\n",
    "ax_t.plot(V, color=\"grey\")\n",
    "ax_t.set_ylim([V_0 * 1.1, np.abs(V_0)/10])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Determination of the Bloch factors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 341,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "dx = 0.01\n",
    "steps_per_well = np.int(1. / dx)\n",
    "potential_width = np.int(0.4 / dx)\n",
    "n_x = 20\n",
    "\n",
    "V_0 = 30\n",
    "V = np.zeros(steps_per_well)\n",
    "n_w = steps_per_well\n",
    "B = np.zeros((n_w, n_w), dtype=np.complex)\n",
    "ks = 1.j * np.linspace(-np.pi, np.pi, n_x)\n",
    "\n",
    "for i in range(1):\n",
    "    left = i * steps_per_well\n",
    "    right = i * steps_per_well + potential_width\n",
    "    V[left:right] = V_0\n",
    "\n",
    "bnd_0 = []\n",
    "bnd_1 = []\n",
    "bnd_2 = []\n",
    "bnd_3 = []\n",
    "\n",
    "for k in ks:\n",
    "    B[:] = 0. + 0.j\n",
    "    \n",
    "    np.fill_diagonal(B, +2 * 1 + dx * dx * (np.abs(k * k) + V))\n",
    "    B = B + np.diag((-1 + k * dx) * np.ones(n_w - 1), -1)\n",
    "    B = B + np.diag((-1 - k * dx) * np.ones(n_w - 1), +1)\n",
    "\n",
    "    # periodic boundary conditions\n",
    "    B[0, -1] = -1 + k * dx\n",
    "    B[-1, 0] = -1 - k * dx\n",
    "    \n",
    "    evals, evecs = la.eigh(B)\n",
    "    bnd_0.append(evals[0] / (dx * dx))\n",
    "    bnd_1.append(evals[1] / (dx * dx))    \n",
    "    bnd_2.append(evals[2] / (dx * dx))    \n",
    "    bnd_3.append(evals[2] / (dx * dx))    \n",
    "\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 344,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "plt.plot(bnd_0)\n",
    "plt.plot(bnd_1)\n",
    "plt.plot(bnd_2)\n",
    "plt.plot(bnd_3)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 345,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "plt.plot(evecs[:, 4])\n",
    "plt.plot(V / 1000)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Wannier functions from Kronig Penney functions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Solve for the Bloch factors**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 349,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "V_0 = 100\n",
    "n_unit_cells = 20\n",
    "a = 2.\n",
    "dx = 0.1\n",
    "steps_per_well = np.int(a / dx)\n",
    "potential_width = np.int(0.2 / dx)\n",
    "\n",
    "n_w = steps_per_well\n",
    "V = np.zeros(steps_per_well)\n",
    "B = np.zeros((n_w, n_w), dtype=np.complex)\n",
    "ks = np.array([1.j * (-np.pi / a + 2 * np.pi / a / n_unit_cells * s) for s in range(n_unit_cells)])\n",
    "\n",
    "boundary = np.int(potential_width / 2)\n",
    "boundary = 2\n",
    "V[0:boundary] = V_0\n",
    "V[-boundary:] = V_0\n",
    "\n",
    "evls = {}\n",
    "evcs = {}\n",
    "\n",
    "for nk, k in enumerate(ks):\n",
    "    B[:] = 0. + 0.j\n",
    "    \n",
    "    np.fill_diagonal(B, 2 * 1 + dx * dx * (np.abs(k * k) + V))\n",
    "    B = B + np.diag((-1 + k * dx) * np.ones(n_w - 1), -1)\n",
    "    B = B + np.diag((-1 - k * dx) * np.ones(n_w - 1), +1)\n",
    "\n",
    "    B[0, -1] = -1 + k * dx\n",
    "    B[-1, 0] = -1 - k * dx\n",
    "    \n",
    "    evals, evecs = la.eigh(B)\n",
    "    evls[nk] = evals / (dx * dx)\n",
    "    evcs[nk] = evecs[:, 0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Build Bloch functions from Bloch factors**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 350,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "psi_ungauged = np.zeros((n_unit_cells, n_unit_cells * steps_per_well), dtype=np.complex)\n",
    "\n",
    "for nk in range(n_unit_cells):\n",
    "    for nx in range(n_unit_cells):\n",
    "        for i in range(steps_per_well):\n",
    "            jx = nx * steps_per_well + i\n",
    "            x = jx * dx\n",
    "            psi_ungauged[nk, jx] = np.exp(ks[nk] * x) * evcs[nk][i] / np.sqrt(n_unit_cells)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Gauge Bloch functions**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 351,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "phases = np.zeros(n_unit_cells, dtype=np.complex)\n",
    "for nk in range(n_unit_cells):\n",
    "    phases[nk] = psi_ungauged[nk, 0] / np.abs(psi_ungauged[nk, 0])\n",
    "    \n",
    "psi_gauged = np.zeros((n_unit_cells, n_unit_cells * steps_per_well), dtype=np.complex)\n",
    "\n",
    "for nk in range(n_unit_cells):\n",
    "    psi_gauged[nk, :] = psi_ungauged[nk, :] / phases[nk]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    " **Show the result**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 356,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "wannier = np.zeros(n_unit_cells * steps_per_well, dtype=np.complex)\n",
    "i = 1\n",
    "\n",
    "for jx in range(n_unit_cells * steps_per_well):\n",
    "    for nk in range(n_unit_cells):\n",
    "        wannier[jx] += np.exp(-ks[nk] * a * 10) * psi_gauged[nk, jx] / np.sqrt(n_unit_cells)\n",
    "        \n",
    "plt.plot(dx * np.arange(n_unit_cells * steps_per_well), np.real(wannier), color=\"grey\")    \n",
    "plt.axvline(x=i * steps_per_well, linewidth=3)\n",
    "plt.axvline(x=(i + 1) * steps_per_well, linewidth=3)\n",
    "# plt.xlim([10, 30])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}