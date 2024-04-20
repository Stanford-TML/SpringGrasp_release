# SpringGrasp
## Synthesizing Compliant, Dexterous Grasp under Shape Uncertainty
Optimization based compliant grasp synthesis using only single depth image.

## Installation
### Install basic python dependencies
```
pip install -r requirements.txt
```

### Install thirdparty tools
```
cd thirdparty
cd differentiable-robot-model
pip install -e .
cd TorchSDF
bash install.sh
```
### Install curobo for arm motion planning and collision avoidance [optional]
```
mkdir curobo_ws
```
Download and unzip customized [curobo](https://drive.google.com/file/d/1uNE-5SKdsH63a3fXlR7KLqrvdTvE27bA/view?usp=sharing) inside `curobo_ws`

Follow install instruction of each package in their `README.md`.

## File structure
```
root directory
  ├── assets  
  │   └── // folders for real scanned object
  ├── data  
  │   └── // folders for data after optimization
  ├── gpis_states  
  │   └── // state data for restoring and visualizing gaussian process implicit surface
  ├── thirdparty  
  │   ├── // dependent third-party package
  |   └── TorchSDF
  |      └── // Compute SDF of mesh in pytorch
  |   └── differentiable-robot-model
  |      └── // Differentiable forward kinematics model
  ├── [curobo_ws] // customized curobo motion planner
  │   ├── curobo
  |   └── nvblox
  |   └── nvblox_torch
  ├── gpis
  |   ├── 3dplot.py // Visualizing GPIS intersection and its uncertainty
  |   └── gpis.py // Definition for Gaussian process implicit surface
  ├── spring_grasp_planner // Core implementation
  |   ├── initial_guesses.py // Initial wrist poses
  |   ├── metric.py // Implementation of spring grasp metric
  |   └── gpis.py // Different optimizers for spring grasp planner
  ├── utils
  |   └── // Ultilities to support the project
  ├── process_pcd.py // Processing pointclouds from different cameras
  ├── optimize_pregrasp.py // Running compliant pregrasp optimization
  └── verify_grasp_robot.py  // verifying pregrasp on hardware, kuka iiwa14 + left allegro hand.
```

## Usage
### Plan grasp for pre-scanned objects
#### Process pointcloud
```
python process_pcd.py --exp_name <obj_name>
```
#### Plan and visualizing grasp
```
python optimize_pregrasp.py --exp_name <obj_name>
```
### Customized object
#### Plan and visualize grasp
```
python optimize_pregrasp.py --exp_name <obj_name> --pcd_file <path to your ply pointcloud>
```
#### Plan reaching motion for Kuka arm (require curobo installation and scene configuration)
```
python traj_gen.py --exp_name <obj_name>
```
#### Execute grasp on real robot(require support for kuka iiwa14 + left allegro hand)
```
python verify_grasp_robot.py --exp_name <obj_name>
```
### Using customized scene and deploy on hardware
How to deploy on hardware varies case by case, if you need help with using Kuka iiwa14 + allegro hand or run into troubles with coordinate system convention please contact: ericcsr [at] stanford [dot] edu