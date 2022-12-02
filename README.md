# Model3d
Scanning 3D model with ArUco Marker

<img src="./docs/pointcloud_output.png" width=30% height=30%> <img src="./docs/Mesh_output.png" width=30% height=30%> <img src="./docs/Mesh_color_output.png" width=30% height=30%>

Require zivid package https://github.com/zivid/zivid-ros



To start zivid two camera
```
roslaunch model3d zivid_run.launch
```
Set environment by put the object on the aruco marker board

<img src="./docs/Capture_process.jpg" width=30% height=30%> <img src="./Board/pic/poseboardA3.png" width=55% height=55%>

To start scanning 3d model by zivid capturing aruco marker board
```
rosrun model3d zivid_model.py
```

# Then press the key to process
    :: Key command ::
c : Capturing & Merge Model

p : Preview pcd_merged

s : Save

e : Shutdown the Server
