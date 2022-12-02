import numpy as np
import open3d as o3d
import zivid
import copy
import msvcrt
import datetime

class sphere:
    def __init__(self, center, color):
        self.pcd = o3d.geometry.TriangleMesh.create_sphere(radius = 0.01)
        self.pcd.compute_vertex_normals()
        self.pcd.translate((center[0], center[1], center[2]), relative=False)
        self.pcd.paint_uniform_color(color)

def capture():
    app = zivid.Application()
    #       For Camera
    camera = app.connect_camera() 
    capture_assistant_params = zivid.capture_assistant.SuggestSettingsParameters(max_capture_time=datetime.timedelta(milliseconds=1200), ambient_light_frequency="hz50",)
    settings = zivid.capture_assistant.suggest_settings(camera, capture_assistant_params)
    frame = camera.capture(settings)
    
    return frame

def FrameToPCD(frame):
    print(":: Convert to PCD....")
    point_cloud = frame.point_cloud()
    xyz = point_cloud.copy_data("xyz")
    rgba = point_cloud.copy_data("rgba")
    xyz = np.nan_to_num(xyz).reshape(-1, 3)/1000
    rgb = rgba[:,:, 0:3].reshape(-1, 3)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255)

    return pcd

def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

def sortpoints(conerPoint):
    a = np.add(conerPoint[0],conerPoint[1])
    b = np.add(conerPoint[2],conerPoint[3])
    sumpoint = np.add(a,b)
    # print("sumpoint : ",sumpoint)
    center = sumpoint/4
    datax = center[0]
    datay = center[1]
    dataz = center[2]
    # print("Center : ",center)
    sortpoints = [0,0,0,0]
    for x in conerPoint:

        if datax > x[0] and datay > x[1]:
            sortpoints[0] = x
        elif datax < x[0] and datay > x[1]:
            sortpoints[1] = x
        elif datax < x[0] and datay < x[1]:
            sortpoints[2] = x
        elif datax > x[0] and datay < x[1]:
            sortpoints[3] = x
    
    # print("Sortpoint")
    # print(sortpoints[0])#cross2
    # print(sortpoints[1])
    # print(sortpoints[2])#cross1
    # print(sortpoints[3])#base
    Nvector = normalvector(sortpoints[3],sortpoints[2],sortpoints[0])
    
    sortpoints.append(np.add(sortpoints[0],Nvector*0.2))
    sortpoints.append(np.add(sortpoints[1],Nvector*0.2))
    sortpoints.append(np.add(sortpoints[2],Nvector*0.2))
    sortpoints.append(np.add(sortpoints[3],Nvector*0.2))
    # print(sortpoints)
    sortpoints = np.array(sortpoints)

    return sortpoints,Nvector

def normalvector(base, cross1, cross2):
    vector1 = np.subtract(cross1,base)
    vector2 = np.subtract(cross2,base)
    normalVec = np.cross(vector1,vector2)
    UNormalVec = normalVec/np.linalg.norm(normalVec)
    print("Normal : ",UNormalVec)
    return UNormalVec

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def cropPCD(pcd,conerPoint):
    # corners = np.array([conerPoint[0],conerPoint[1],conerPoint[2],conerPoint[3]
    #                 ,conerPoint[4],conerPoint[5],conerPoint[6],conerPoint[7]])
    conerPoint = np.array(conerPoint)
    corners,Nvector = sortpoints(conerPoint)
    rotation = rotation_matrix_from_vectors([0,0,1], Nvector)

    # box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(corners))
    # box = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(box)
    # box.rotate(rotation,box.center)


    box = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(corners))
    box.color = (1, 0, 0)
    box.translate( Nvector*0.005,relative=True)

    sphere1 = sphere(corners[0], (0,0.5,0.5))
    sphere2 = sphere(corners[1], (0,0.5,0.5))
    sphere3 = sphere(corners[2], (0,0.5,0.5))
    sphere4 = sphere(corners[3], (0,0.5,0.5))
    sphere5 = sphere(corners[4], (0,0.5,0.5))
    sphere6 = sphere(corners[5], (0,0.5,0.5))
    sphere7 = sphere(corners[6], (0,0.5,0.5))
    sphere8 = sphere(corners[7], (0,0.5,0.5))

    croped = pcd.crop(box)
    o3d.visualization.draw_geometries([Realcoor,box,croped,sphere1.pcd,sphere2.pcd,sphere3.pcd,sphere4.pcd,sphere5.pcd,sphere6.pcd,sphere7.pcd,sphere8.pcd])
    return box

def setworkspace():
    app = zivid.Application()
    camera = app.connect_camera() 
    capture_assistant_params = zivid.capture_assistant.SuggestSettingsParameters(max_capture_time=datetime.timedelta(milliseconds=1200), ambient_light_frequency="hz50",)
    settings = zivid.capture_assistant.suggest_settings(camera, capture_assistant_params)

    frame = camera.capture(settings)
    Workspace = FrameToPCD(frame)
    PointWS = pick_points(Workspace)
    pointcloud_as_array = np.asarray(Workspace.points)
    coners = []
    for i,x in enumerate(PointWS):
        coners.append(pointcloud_as_array[x])
    box = cropPCD(Workspace,coners)
    return box

fixbox = o3d.geometry.OrientedBoundingBox()
Realcoor = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1,[0,0,0])
if __name__ == '__main__':

    i = 8
    while True:
        print("=================================================================================")
        print("\n:: Key command ::\n\tc : Capture the point cloud and process\n\tw : Setworkspace\n\ts : Save\n\te : Shutdown the Server")
        key = msvcrt.getch().lower()
        if key == b'c':
            print("\n:: Capturing the Pointcloud")
            frame = capture()
            pcd = FrameToPCD(frame)
            if not fixbox.is_empty():
                pcd = pcd.crop(fixbox)
            o3d.visualization.draw_geometries([Realcoor,pcd])
        elif key == b'w':
            fixbox = setworkspace()
        elif key == b's':
            o3d.io.write_point_cloud(f"D:/Oongking/iRAP/Robot arm/program/ZED2/buildModel/data/fan/fan_fragment_Onfoam{i}.pcd", pcd)
            print("complete")
            i+=1
        elif key == b'e':
            break
        
        else:
            print("Wrong Key")
    print("\n------ Shutdown the Server ------ ")

