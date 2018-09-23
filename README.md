# Rigorous-Coupled-Wave-Analysis PACKAGE

semi-analytic fourier series solutions for Maxwell's equations. This package actually contains three methods:
1) TMM: classical transfer matrix method
2) Plane Wave Expansion Method: essentially solves Maxwell's equations in fourier space.
3) RCWA: a synthesis of TMM and PWEM.

## USAGE
Right now, the package is partitioned by the type of numerical method used. Examples for each method are in the folder "method_name"_examples

## First TMM
TMM is usually the 'intellectual' precursor to RCWA. It assumes different layers but each layer has no structure (could be anisotropic however)
A simple demonstration of TMM can be done by reproducing the spectrum of a Bragg grating.
![Alt Text](./img/bragg_TMM.png)

## PWEM
Plane wave expansion method is simply the solution of Maxwell's equations in Fourier space.

## Transitioning to RCWA
RCWA is a synthesis of TMM and PWEM in that it targets layered structures whereby each layer contains some sort of periodic pattern

![Alt Text](./img/sample_1D_grating_spectra.png)
