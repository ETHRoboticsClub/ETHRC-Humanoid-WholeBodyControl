# Installation 

## For the first time : 
```shell
./docker/run_docker.sh --install --root
```

## Re enter the docker : 
```shell
./docker/run_docker.sh --root
```

## Installation of the robocasa objects 
```shell
python -m decoupled_wbc.dexmg.gr00trobocasa.robocasa.scripts.setup_macros

python -m decoupled_wbc.dexmg.gr00trobocasa.robocasa.scripts.download_kitchen_assets
```

# Run the script 

## Without teleop : 
```shell

python decoupled_wbc/scripts/deploy_g1.py     --interface sim     --camera_host localhost     --sim_in_single_process     --simulator robocasa     --image-publish     --enable-offscreen     --env_name PnPBottleRomain    
```

## With teleop : 
On robot PC, double click app icon of XRoboToolkit-PC-Service or run service
```shell 
    /opt/apps/roboticsservice/runService.sh
```

```shell
python decoupled_wbc/scripts/deploy_g1.py     --interface sim     --camera_host localhost     --sim_in_single_process     --simulator robocasa     --image-publish     --enable-offscreen     --env_name PnPBottleRomain     --hand_control_device=pico     --body_control_device=pico
```
(the name of the env must be changed)


### Task prompt for the PnPBottleRomain task. 

Move to the right first, then pick up the cardboard box located on the countertop in front of the refrigerator, and place it on the right side of the sink.


### Task prompt for the PickPlaceBottleLoco task

Turn right, move in front of the water bottle on the counter in front of the fridge, pick up the water bottle, then turn left, walk to the sink, and place the water bottle into the sink.


### Trim the dataset if needed : 
```shell
conda activate lerobot-trim
```
Add rights to modify the dataset (because was created from docker container) : 
```shell 
sudo chown rguntz /home/rguntz/Desktop/humanoid/ETHRC-Humanoid-WholeBodyControl/outputs/
```
 
Then run the dashboard : 
```shell
cd /home/rguntz/Desktop/humanoid/ETHRC-Humanoid-WholeBodyControl
python trim_dashboard/app.py
``

