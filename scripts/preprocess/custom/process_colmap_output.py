import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import open3d as o3d

WORKSPACE_DIR = "./datafiles/custom/triangulation"
FOCAL_LENGTH = 768
RESOLUTION_X = 640.0
RESOLUTION_Y = 360.0

def convert_quaternion_to_RT_matrix(r,t):
    correction = np.array(([1.0, 1.0, 1.0, 1.0],[-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]))  
    M = np.empty((4, 4))
    M[:3, :3] = r
    M[:3, 3] = t/1000
    M[3, :] = [0.00, 0.00, 0.00, 1.00]
    M = M*correction
    return M.flatten()

def sort_data_by_frame_index(lines, output_filename):
    
    # Strip lines of trailing white space, ignore the initial index, and split on spaces
    lines = [line.strip().split(' ', 1)[1] for line in lines]

    # Sort lines by the filename which is the last element when split by space
    lines.sort(key=lambda line: line.split()[-1])

    # Add new indices to the sorted lines
    lines = [f"{i+1} {line}" for i, line in enumerate(lines)]
    with open(output_filename, 'w') as f:
        f.write('\n'.join(lines))
    return lines

if __name__ == '__main__':
    
    # Creating sorted camera pose matrix
    with open(os.path.join(WORKSPACE_DIR,'images.txt'), 'r') as fin:
        data = fin.read().splitlines(True)
    sort_data_by_frame_index(data[4::2], os.path.join(WORKSPACE_DIR,'images_sorted.txt'))
    data = np.transpose(np.loadtxt(os.path.join(WORKSPACE_DIR,'images_sorted.txt'), skiprows=0, delimiter=" ", dtype={'names': ('IMAGE_ID', 'QW', 'QX', 'QY', 'QZ', 'TX', 'TY', 'TZ', 'CAMERA_ID', 'NAME'),'formats': (np.int16, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.int16, '|S15')}))
    with open(os.path.join(WORKSPACE_DIR,'custom.matrices.txt'), 'a') as fout:
        for line in range(len(data)):
            t = np.array([data[line][5],data[line][6],data[line][7]])
            r = R.from_quat(np.array([data[line][1],data[line][2],data[line][3],data[line][4]])).as_matrix()
            RT = convert_quaternion_to_RT_matrix(r, t)
            fout.writelines(str(line) + " " + np.array2string(RT[:12], formatter={'float_kind':lambda x: "%.17f" % x}, separator=' ', max_line_width=1000)[1:-1] + " " + np.array2string(RT[12:], formatter={'float_kind':lambda x: "%.1f" % x}, separator=' ', max_line_width=1000)[1:-1] + "\n")
    
    # Creating camera intrinsic matrix
    focal_length = 768
    res_x = 640.0
    res_y = 360.0
    cam_params = np.array([FOCAL_LENGTH, FOCAL_LENGTH, RESOLUTION_X/2, RESOLUTION_Y/2])
    with open(os.path.join(WORKSPACE_DIR,'custom.intrinsics.txt'), 'a') as fout:
        for line in range(len(data)):
            t = np.array([data[line][5],data[line][6],data[line][7]])
            r = R.from_quat(np.array([data[line][1],data[line][2],data[line][3],data[line][4]])).as_matrix()
            RT = convert_quaternion_to_RT_matrix(r, t)
            fout.writelines(str(line) + " " + np.array2string(cam_params, formatter={'float_kind':lambda x: "%.17f" % x}, separator=' ', max_line_width=1000)[1:-1] + "\n")
    
    # Converting .ply to .obj
    os.path.join(WORKSPACE_DIR,'custom.ply')
    mesh = o3d.io.read_triangle_mesh(os.path.join(WORKSPACE_DIR,'custom.ply'))
    o3d.io.write_triangle_mesh(os.path.join(WORKSPACE_DIR,'custom.obj'),mesh)
