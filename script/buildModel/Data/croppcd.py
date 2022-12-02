import numpy as np
import open3d as o3d
import copy

def fixbox(rot,trans,z_offset) :
    # Before rotate to canon
    y = 0.125
    x = 0.125
    z = 0.1
    
    fix_box = np.array([
    [-x/2,-y/2,z_offset],
    [-x/2, y/2,z_offset],
    [x/2, y/2,z_offset],
    [x/2,-y/2,z_offset],

    [-x/2,-y/2,z+z_offset],
    [-x/2, y/2,z+z_offset],
    [x/2,-y/2,z+z_offset],
    [x/2, y/2,z+z_offset]
    ])

    fixbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(fix_box))
    fixbox.rotate(rot,(0,0,0))
    fixbox.translate(np.asarray(trans,dtype=np.float64),relative=True)
    fixbox.color = (0, 0, 0)

    return fixbox 


transformation_matrix = np.array([  [1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]],
                                    dtype=float)

z_offset = -0.01
box = fixbox(transformation_matrix[:3, :3],transformation_matrix[ 3, :3],z_offset)
pcd = o3d.io.read_point_cloud("/home/oongking/RobotArm_ws/src/model3d/script/buildModel/Data/FanCoverTop.pc")

Realcoor = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01,[0,0,0])

o3d.visualization.draw_geometries([Realcoor,pcd,box])

croppcd = pcd.crop(box)
o3d.visualization.draw_geometries([Realcoor,croppcd,box])

# o3d.io.write_point_cloud(f"/home/oongking/RobotArm_ws/src/model3d/script/buildModel/Data/FanCoverTop.pcd", croppcd)