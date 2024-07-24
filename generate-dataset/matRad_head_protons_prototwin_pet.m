%% Example: Proton Treatment Plan for the Head Plans in PROTOTWIN-PET
%er
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright 2017 the matRad development team. 
% 
% This file is part of the matRad project. It is subject to the license 
% terms in the LICENSE file found in the top-level directory of this 
% distribution and at https://github.com/e0404/matRad/LICENSES.txt. No part 
% of the matRad project, including this file, may be copied, modified, 
% propagated, or distributed except according to the terms contained in the 
% LICENSE file.
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% matRadGUI

%%
% In this example we will show 
% (i) how to load patient data into matRad
% (ii) how to setup a proton dose calculation 
% (iii) how to inversely optimize the pencil beam intensities directly from command window in MATLAB. 
%% Patient Data Import
% Let's begin with a clear Matlab environment and import the data

matRad_rc; %If this throws an error, run it from the parent directory first to set the paths
load HN-CHUM-018.mat;  % selected patient from HEAD-NECK-PET-CT dataset
% load HEAD_AND_NECK.mat;  % selected patient from CORT dataset
% matRadGUI

% cst(36, :) = [];  % to remove couch in order to add a posterior beam (from below the couch)
% body_idcs = cst(5,4);
% body_idcs = body_idcs{1, 1};
% body_idcs = body_idcs{1};
% ct_cube = ct.cubeHU{1};
% % make everything outside the patient equal to air
% total_voxels = numel(ct_cube); % Total number of elements
% all_indices = 1:total_voxels; % Linear indices for the entire ct
% % Find indices that are not in the provided 'indices'
% indices_to_set = setdiff(all_indices, body_idcs); 
% ct_cube(indices_to_set) = -1000;  % set to air
% 
% ct.cubeHU{1} = ct_cube;
%% Treatment Plan
% The next step is to define your treatment plan labeled as 'pln'. This 
% structure requires input from the treatment planner and defines the most 
% important cornerstones of your treatment plan.

pln.radiationMode = 'protons';        
pln.machine       = 'Generic';  
% Can choose between Generic (/basedata/protons_Generic.mat) (>30MeV) and
% generic_MCsquare (/basedata/protons_generic_MCsquare.mat) machines
% (>70MeV)

%%
% Define the flavor of biological optimization for treatment planning along
% with the quantity that should be used for optimization. As we use 
% protons, we follow here the clinical standard and use a constant relative
% biological effectiveness of 1.1. Therefore we set bioOptimization to 
% const_RBExD
pln.propOpt.bioOptimization = 'const_RBExD';

%%
% for particles it is possible to also calculate the LET disutribution
% alongside the physical dose. To activate the corresponding option during 
% dose calculcation set to 1. Otherwise, set to 0
pln.propDoseCalc.calcLET = 0;
                                       
%%
% Now we have to set the remaining plan parameters.
pln.numOfFractions        = 30;
pln.propStf.gantryAngles  = [75 -75]; % 180];
pln.propStf.couchAngles   = [-15 15]; % 0];
pln.propStf.bixelWidth    = 5;  % Standard protons
pln.propStf.numOfBeams    = numel(pln.propStf.gantryAngles);
pln.propStf.isoCenter     = ones(pln.propStf.numOfBeams,1) * matRad_getIsoCenter(cst,ct,0);
pln.propOpt.runDAO        = 0;
pln.propOpt.runSequencing = 0;

% dose calculation settings
pln.propDoseCalc.doseGrid.resolution.x = 5; % [mm]
pln.propDoseCalc.doseGrid.resolution.y = 5; % [mm]
pln.propDoseCalc.doseGrid.resolution.z = 5; % [mm]

%% Generate Beam Geometry STF
stf = matRad_generateStf(ct,cst,pln);

%% Dose Calculation
% Lets generate dosimetric information by pre-computing dose influence 
% matrices for unit beamlet intensities. Having dose influences available 
% allows for subsequent inverse optimization. 
dij = matRad_calcParticleDose(ct,stf,pln,cst);

%% Inverse Optimization for IMPT
% The goal of the fluence optimization is to find a set of bixel/spot 
% weights which yield the best possible dose distribution according to the 
% clinical objectives and constraints underlying the radiation treatment
resultGUI = matRad_fluenceOptimization(dij,cst,pln);

%% Open the GUI
matRadGUI;

%% Save the output
row_body_indices = find(strcmpi(cst(:,2), 'body') | strcmpi(cst(:,2), 'skin'));
body_indices = cst{row_body_indices, 4}{1, 1};
CT_cube = ct.cubeHU{1, 1};
CT_resolution = [ct.resolution.x, ct.resolution.y, ct.resolution.z];
CT_offset = ct.dicomInfo.ImagePositionPatient;
weights = resultGUI.w;
load protons_Generic.mat
machine_data = machine.data;
dataToSave = {'ct', 'CT_cube', 'CT_resolution', 'CT_offset', 'cst', 'body_indices', 'stf', 'weights', 'machine_data'};
%%% HN-CHUM-018
output_folder_pc = '/path/to/HeadPlans/HN-CHUM-018';
output_folder_repo = '/path/to/prototwin-pet/data/HN-CHUM-018';

%%% Head and Neck CORT
% output_folder_pc = '/path/to/head-cort';
% output_folder_repo = '/path/to/prototwin-pet/data/head-cort';

save(fullfile(output_folder_repo, 'matRad-output.mat'), dataToSave{:});
save(fullfile(output_folder_pc, 'matRad-output.mat'), dataToSave{:});
