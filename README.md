# ADL-dataset

This repo is used for collecting a turntable dataset with ROS and Microsoft Kinect 1

## Build

To build the data collecting software in ROS, build your [workspace](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment):
```Shell
  mkdir -p ~/ADL_dataset_ws/src
  cd ADL_dataset_ws
  catkin_make
  ```
Then clone the repository in your ROS workspace
```Shell
  cd src/
  git clone -r https://github.com/WanlinYang/ADL-dataset.git
  ```
From the base directory of the worksapce, build the code
```Shell
  cd ~/ADL_dataset_ws
  catkin_make
  ```
