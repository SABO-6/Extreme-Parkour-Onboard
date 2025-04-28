# Deployment Code of Extreme Parkour on Unitree Go2

This repository provides an **unofficial implementation** for deploying the project [Extreme Parkour with Legged Robots](https://github.com/chengxuxin/extreme-parkour) on the **Unitree Go2** quadrupted robot. The original work was developed for A1 robots and does not provide the deployment code. 

## Key Contributions

- Add detailed comments throughout the training code. Documented the previously unexplained **observation** vector.

- Add camera randomization during training to accout for Go2's movable camera (unlike A1's fixed camera).

- Provide trained weights and deployment code for Unitree Go2.

## Deployment Instructions

#### Environment Setup
Make sure the environment is properly set up on your Go2 robot, including rclpy, torch and unitree sdk. (If you need guidance for the environment setup, feel free to open an issue - I will provide detailed instructions.)

#### Hardware Setup
Install the **Intel RealSense D435i** depth camera on the Go2. Verify that the captured images resemble the simulation (can be checked using `rviz`).

#### Deployment Steps
1. Connect to the Go2 robot wirelessly via SSH (wired connection is also ok).
```bash
ssh unitree@<go2_ip_address>
```

2. In the first terminal, start the visual node:
```bash
python3 visual_extreme_parkour.py --logdir traced
```

This script retrieves depth images from the D435i camera and publish them at 100Hz to the appropriate ROS topic.

3. In a second terminal, start the controller node:
```bash
python3 run_extreme_parkour.py --logdir traced
```
This script fuses the depth image and proprioception data. Now press the **L1 button** on the remote controller to execute the policy (the robot will process actions but will **not walk yet**).

4. Prepare for movement:
- Turn off the builtin sport service.
- Place the robot in a free-hanging pose (e.g., in a small box or lifted by hand), allowing it to perform motions safely.

5. Re-run the behavior policy node to allow walking:
```bash
python3 run_extreme_parkour.py --logdir traced --nodryrun
```
After pressing **L1**, the robot should now execute the walking strategy. To stop it, press **L2** or **R2**.

## Notes and Tips

#### Policy selection:
Modify in `run_extreme_parkour.py`:
```bash
base_model = 'your_base_model.pth'
vision_model = 'your_vision_model.pth'
```

#### Parkour Mode:
Adjust the one-hot encoding in `unitree_ros2_real.py` (line 472) to switch between **walking** and **parkour** mode:
```bash
parkour_walk = torch.tensor([[1, 0]], device=self.model_device, dtype=torch.float32)
```
Since the original work trained separate policies for these two tasks.

#### Performance disclaimer
The provided weights achieve robust walking and minor obstacle negotiation. However, **parkour performance** is still noticeably below that shown in the original paper. Differences in depth camera behavior (fixed vs mobile, sim vs real) may be a major cause. 


## Acknowledgments
This repository is based on modification of [Robot Parkour Learning](https://github.com/ZiwenZhuang/parkour). Special thanks to the original authors for their open-source contribution.

## Contact
I am a beginner in robotics, and warmly welcome feedback and contributions to improve this repository. For questions, suggestions or collaboration, please open an issue or contact me directly.
