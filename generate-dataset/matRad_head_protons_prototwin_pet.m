%% Example: Proton Treatment Plan for the Head Plans in PROTOTWIN-PET
%
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

% matRadGUI  % uncomment to open the GUI and create the patient folder

%% Patient Data Import
% Let's begin with a clear Matlab environment and import the data

matRad_rc; %If this throws an error, run it from the parent directory first to set the paths

%patient_name = 'HN-CHUM-018';
%load HN-CHUM-018.mat
patient_name = 'prostate-cort';
load PROSTATE.mat;  % selected patient from CORT dataset

couch_idx = find(strcmpi(cst(:,2), 'COUCH'));
cst(couch_idx, :) = [];  % to remove couch in order to add a posterior beam (from below the couch)
row_body_indices = find(strcmpi(cst(:,2), 'body') | strcmpi(cst(:,2), 'skin') | strcmpi(cst(:,2), 'CONTOUR EXTERNE'));  % body is sometimes called skin or contour externe
body_indices = cst{row_body_indices, 4}{1, 1};
ct_cube = ct.cubeHU{1};
% make everything outside the patient equal to air
total_voxels = numel(ct_cube); % Total number of elements
all_indices = 1:total_voxels; % Linear indices for the entire ct
% Find indices that are not in the provided 'indices'
indices_to_set = setdiff(all_indices, body_indices); 
ct_cube(indices_to_set) = -1024;  % set to air
ct.cubeHU{1} = ct_cube;

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
pln.propStf.gantryAngles  = [90];
pln.propStf.couchAngles   = [0];
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

%% Save the output
names = cst(:,2);
% Find names that include 'CTV' (case-insensitive)
matches = cellfun(@(x) contains(x, 'CTV', 'IgnoreCase', true), names);
% Get the row indices where the names include 'CTV'
row_CTV_indices = find(matches);
% Extract CTV_indices from cst{row_CTV_indices,4}{1,1}
CTV_indices_cells = cellfun(@(x) x{1,1}, cst(row_CTV_indices,4), 'UniformOutput', false);
CTV_indices = cell2mat(CTV_indices_cells);
% Get unique CTV_indices
CTV_indices = unique(CTV_indices);

CT_cube = ct.cubeHU{1, 1};
CT_resolution = [ct.resolution.x, ct.resolution.y, ct.resolution.z];
CT_offset = ct.dicomInfo.ImagePositionPatient;
weights = resultGUI.w;
load protons_Generic.mat
machine_data = machine.data;
dataToSave = {'ct', 'CT_cube', 'CT_resolution', 'CT_offset', 'cst', ...
    'body_indices', 'CTV_indices', 'stf', 'weights', 'machine_data'};

output_folder_repo = sprintf('./data/%s', patient_name);

if ~exist(output_folder_repo, 'dir')
    mkdir(output_folder_repo);
end
save(fullfile(output_folder_repo, 'matRad-output.mat'), dataToSave{:});