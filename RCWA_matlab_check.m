

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% illustration/comparison against dielectric grating with metal spacers
% Note we do not modify the metal spacers, ONLY THE AIR SLITS
% we want to minimize loss using the dielectric, which means the phenomenon
% has to be 'dielectric', not 'plasmonic'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
clear; 
clc;
tic
time1=fix(clock);                   % start time, [sec]
c0 = 299792458;                   % speed of light in vacuum, [m/s]
e0=8.854187817e-12;         % vacuum permittivity 
mu0=pi*4e-7;                        % vacuum permeability
%w_scan = linspace(1e14, 3e15, 100);
%lambda_scan = 1e6*2*pi*c0./w_scan;
theta =(45)*pi/180;                % angle of incidence, [rad]
lambda_scan = linspace(0.5,2, 300);

figure()
band_structure = [];
theta = 0;
slit_width = 0;
% material structure specifications    
air_width = slit_width;
% some indication that smaller lattice constants work better...
dielectric_width = 0.25;
lattice_constant = 0.5;
d = [0.2];                        % thickness of each layer [um] 
N = length(d);                      % # of layers
Period(1:N) = lattice_constant;                % Period [um]

f{1} = [dielectric_width ...
    ]/lattice_constant;
Num_ord = 6;                  % number for the highest diffraction order

epsilon_tracker = [];
for i =1:length(lambda_scan)
    lambda = lambda_scan(i);

    wn=1e4/lambda;                  % wavenumber, [cm-1]
    w = 1e6*2*pi*c0./lambda; % angular frequency, [rad/s]
    k0 = 2*pi/lambda;               %wavevector

    epsilon = 1;
    epsilon_groove =  12;
    %==========================================
         e(1) = 1;   % Usually is air or vacuum 
         % Layered structure

          e_m{1}(1) = epsilon ;        % Ridge material (dielectric)  
          e_m{1}(2) = epsilon_groove;                   % Groove material (metal)

          %Substrate
          e(2)= 1+1i*1e-12;         %  air or opaque substrate...does not help reflection
          [Ref(i), Tran(i)] = RCWA_Multi_TM(N, e_m, e_m, e_m, f, Period, d, e, lambda, theta, Num_ord); 
    %==========================================
    %Ref(i)

end

plot(lambda_scan,Ref);
hold on;
plot(lambda_scan, Tran);
plot(lambda_scan, 1-Ref-Tran, '--');
legend('reflection',  'transmission', 'absorption')
drawnow();
hold on;
xlabel('wavelength (microns)')
ylabel('reflectivity')
title(strcat('spectra forhybrid grating_angle=', num2str(theta)))
drawnow()



