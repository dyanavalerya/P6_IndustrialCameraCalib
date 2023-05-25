import open3d as o3d
import numpy as np
from math import radians, cos, sin, sqrt
import glob
import matplotlib.pyplot as plt

def fit_plane_ransac(pcd, source):
    if source == "cad":
        # Settings for the mesh down-sampled pyramid
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                    ransac_n=3,
                                                    num_iterations=1000,
                                                    probability=0.9999)
    elif source == "proposed_sol":
        # Settings for the big pyramid 340 x 160 mm
        plane_model, inliers = pcd.segment_plane(distance_threshold=1,
                                                 ransac_n=3,
                                                 num_iterations=100000,
                                                 probability=1.0)
    elif source == "current_sol":
        # Settings for the big pyramid 340 x 160 mm
        plane_model, inliers = pcd.segment_plane(distance_threshold=10,
                                                 ransac_n=3,
                                                 num_iterations=100000,
                                                 probability=1.0)
    # Settings for the small pyramid 170 x 40 mm
    # plane_model, inliers = pcd.segment_plane(distance_threshold=1.2,
    #                                             ransac_n=3,
    #                                             num_iterations=100000,
    #                                             probability=1.0)

    [a, b, c, d] = plane_model.numpy().tolist()
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud = inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # o3d.visualization.draw_geometries([inlier_cloud.to_legacy(), outlier_cloud.to_legacy()],
    #                                   zoom=0.8,
    #                                   front=[-0.4999, -0.1659, -0.8499],
    #                                   lookat=[2.1813, 2.0619, 2.0999],
    #                                   up=[0.1204, -0.9852, 0.1215])
    return plane_model, outlier_cloud

def dist_k_nearest(pcd_numpy, temp_corner_pt):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pcd_numpy)
    pcd_tree = o3d.geometry.KDTreeFlann(cloud)
    # This one finds one closest neighbour
    [k, idx, _] = pcd_tree.search_knn_vector_3d(temp_corner_pt, 1)
    nearest_pt = pcd_numpy[idx][0]
    return np.linalg.norm(temp_corner_pt - nearest_pt)

def get_top_pt(pcd, source):
    copy_pcd = np.copy(pcd.point.positions.numpy())
    # Fit planes to each of the pyramid sides
    planes_list = np.zeros((4, 4), dtype=np.float64)

    for i in range(4):
        [plane_model, outlier] = fit_plane_ransac(pcd, source)
        pcd = outlier
        planes_list[i] = plane_model.numpy()

    # Make four systems of equations based on the four possible plane intersections
    A = planes_list[0:3, 0:3]
    b = -planes_list[0:3, 3]

    A2 = planes_list[[0, 1, 3], 0:3]
    b2 = -planes_list[[0, 1, 3], 3]

    A3 = planes_list[[0, 2, 3], 0:3]
    b3 = -planes_list[[0, 2, 3], 3]

    A4 = planes_list[[1, 2, 3], 0:3]
    b4 = -planes_list[[1, 2, 3], 3]

    linear_sys = [[A, b], [A2, b2], [A3, b3], [A4, b4]]

    # Solve a linear equation Ax = b, where A = [aix + biy + ciz], b = [di]
    top_pts = np.zeros((4, 3), dtype=np.float64)
    for i, sys in enumerate(linear_sys):
        top_pts[i] = np.linalg.solve(sys[0], sys[1])

    # Average the result of the 4 combinations of plane intersections to get the top corner point
    top_pt = [sum(x) / len(x) for x in zip(*top_pts)]

    # For debugging and visualization purposes
    # top_pts_cloud = o3d.geometry.PointCloud()
    # top_pts_cloud.points = o3d.utility.Vector3dVector(top_pts)
    # top_pts_cloud.paint_uniform_color([1, 0, 0])  # red
    # top_pt_cloud = o3d.geometry.PointCloud()
    # five_top_pts = np.zeros((5, 3), dtype=np.float64)
    # five_top_pts[0:4] = top_pts
    # five_top_pts[4] = np.asarray(top_pt)
    # top_pt_cloud.points = o3d.utility.Vector3dVector(five_top_pts)
    # top_pt_cloud.paint_uniform_color([0, 1, 0])  # green
    #
    # copy_pcd_cloud = o3d.geometry.PointCloud()
    # copy_pcd_cloud.points = o3d.utility.Vector3dVector(copy_pcd)
    #
    # mat = o3d.visualization.rendering.MaterialRecord()
    # mat.shader = 'defaultUnlit'
    # mat.point_size = 20.0
    # o3d.visualization.draw([{'name': 'pcd', 'geometry': top_pt_cloud, 'material': mat}, top_pts_cloud, copy_pcd_cloud],
    #                        show_skybox=False)
    return top_pt, planes_list

def get_plane_pair_combos(planes_list):
    test = np.zeros((6, 2), dtype=np.float64)
    plane_pair = np.zeros((6, 2, 4), dtype=np.float64)
    for i in range(4):
        for j in range(i+1, 4):
            plane_pair[i+j-(1*(i < 1))] = [planes_list[i], planes_list[j]]
            test[i+j-(1*(i < 1))] = [i, j]
    return plane_pair

def plot_line_fit_and_cloud(copy_pcd, vr, vb):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(copy_pcd[:, 0], copy_pcd[:, 1], copy_pcd[:, 2])
    l = [-500, 500]
    lp1 = vr * l[0] + vb
    lp2 = vr * l[1] + vb
    ax.plot([lp1[0], lp2[0]], [lp1[1], lp2[1]], [lp1[2], lp2[2]], 'k')
    # plt.show()

def find_pyramid_base_plane(control_points):
    # For finding the right base corner points, fit a plane to minimum 4 points
    # This plane can be fit only to the base points that form a square
    # The rest of the points coming from the two horizontal lines as results of
    # opposite plane intersections will be removed
    base_cloud = o3d.geometry.PointCloud()
    base_cloud.points = o3d.utility.Vector3dVector(control_points[1:7, :])
    plane_model, inliers = base_cloud.segment_plane(distance_threshold=50,
                                             ransac_n=4,
                                             num_iterations=100000,
                                             probability=1.0)

    inlier_cloud = base_cloud.select_by_index(inliers)
    inlier_cloud = inlier_cloud.paint_uniform_color([0, 1.0, 0])
    outlier_cloud = base_cloud.select_by_index(inliers, invert=True)
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
    #                                   zoom=0.8,
    #                                   front=[-0.4999, -0.1659, -0.8499],
    #                                   lookat=[2.1813, 2.0619, 2.0999],
    #                                   up=[0.1204, -0.9852, 0.1215])
    return inlier_cloud

def get_control_points(pcd, source):
    # Get a copy of the point cloud before it gets altered by the plane fitting process
    # It will be needed later
    copy_pcd = np.copy(pcd.point.positions.numpy())

    # Compute the top point of the pyramid
    top_pt, planes_list = get_top_pt(pcd, source)

    # Make all combinations of plane pairs, as I do not know which ones are opposite or not
    plane_pair = get_plane_pair_combos(planes_list)

    # Initialize the variable that stores the control points
    # It has 7 points because it will contain all the plane pairs
    # intersection lines and the resulting points on them
    # The 7 points will be filtered down to 5 in later processes
    control_points = np.zeros((7, 3), dtype=np.float64)
    control_points[0, :] = top_pt

    count = 0
    # REMEMBER TO CHANGE THIS when changing size of pyramid!!!
    edge_length = 288.8 # mm
    for i, pair in enumerate(plane_pair):
        vb = top_pt
        # Determine the direction vector of the line
        # That is the cross product of the planes' normals, n = [a, b, c]
        n1, n2 = np.array(pair[0][:3]), np.array(pair[1][:3])
        vr = np.cross(n1, n2)
        # Find t such that the 2-norm of the direction vector vr * t = length of the pyramid edge
        t = edge_length / sqrt(pow(vr[0], 2) + pow(vr[1], 2) + pow(vr[2], 2))
        temp_line_eq = vr * t + vb

        # Check if direction of the line was correct
        # If it is not correct, the detected point is lying above the pyramid
        # and does not have any close neighbours
        dist = dist_k_nearest(copy_pcd, temp_line_eq)
        # If the neighbour is located too far, the distance becomes large
        if dist > 10.0:
            # reverse the direction of the line
            temp_line_eq = -vr * t + vb
            # check again if distance gets smaller
            dist2 = dist_k_nearest(copy_pcd, temp_line_eq)
            min_dist = min(dist, dist2)
            # If not, then keep the previous line equation
            if min_dist == dist:
                temp_line_eq = vr * t + vb

        control_points[count+1, :] = temp_line_eq

        # Plot cloud and lines to see if they correspond to the edges
        plot_line_fit_and_cloud(copy_pcd, vr, vb)

        count += 1

    # Remove the points that are not part of the base plane
    inlier_cloud = find_pyramid_base_plane(control_points)

    # Add both the inliers and the top point
    control_points = np.append(control_points[0:1, :], np.asarray(inlier_cloud.points), axis=0)
    return control_points

def hidden_pts_removal(pcd):
    # Remove base point cloud to correspond to real measurement
    diameter = np.linalg.norm(
        np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    radius = diameter.numpy() * 100

    camera = o3d.core.Tensor(np.asarray([187, 400, 200], dtype=np.float32), o3d.core.float32)
    _, pt_map = pcd.hidden_point_removal(camera, radius)
    pcd = pcd.select_by_index(pt_map)

    print("Visualize hidden points removal result")
    # o3d.visualization.draw_geometries([pcd.to_legacy()])

    return pcd

def rotate(alpha=0, beta=0, gamma=0):
    alpha = radians(alpha)
    beta = radians(beta)
    gamma = radians(gamma)
    return np.asarray([[cos(beta)*cos(gamma), -cos(beta)*sin(gamma), sin(beta)],
            [sin(alpha)*sin(beta)*cos(gamma) + cos(alpha)*sin(gamma), -sin(alpha)*sin(beta)*sin(gamma) + cos(alpha)*cos(gamma), -sin(alpha)*cos(beta)],
            [-cos(alpha)*sin(beta)*cos(gamma) + sin(alpha)*sin(gamma), cos(alpha)*sin(beta)*sin(gamma)+sin(alpha)*cos(gamma), cos(alpha)*cos(beta)]])

def rigid_transform_3D(A, B):
    """
    Least-squares fitting of two 3D point sets
    :param A: 3xN matrix of points to be corrected
    :param B: 3xN matrix of points used to correct with
    :return: 3x3 Rotation matrix and 3x1 translation vector from A to B
    """
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    # To get transformation from A to B
    H = Am @ np.transpose(Bm)
    # To get transformation from B to A
    # H = Bm @ np.transpose(Am)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B
    # t = -R @ centroid_B + centroid_A

    return R, t

if __name__ == '__main__':

    # The automated way (the commented code) does not work well with the transformation,
    # but it detects control points and visualizes them, so you can use it for that

    # print('Reading inputs')
    # mat = o3d.visualization.rendering.MaterialRecord()
    # mat.shader = 'defaultUnlit'
    # mat.point_size = 10.0

    # proposed_sol_calib_pts = []
    # for each_file in glob.glob('proposed_sol/*.ply'):
    #     proposed_sol_cloud = o3d.t.io.read_point_cloud(each_file)
    #     control_pts = get_control_points(proposed_sol_cloud, "proposed_sol")
    #
    #     # Check if control points are found right
    #     proposed_sol_cloud.paint_uniform_color([0.1, 0.1, 0.1])
    #     control_pts_cloud = o3d.geometry.PointCloud()
    #     control_pts_cloud.points = o3d.utility.Vector3dVector(control_pts)
    #     control_pts_cloud.paint_uniform_color([0, 1, 0])  # green
    #     # See if detected points lie in the right locations
    #     o3d.visualization.draw([{'name': 'pcd', 'geometry': control_pts_cloud, 'material': mat}, proposed_sol_cloud.to_legacy()],
    #                            show_skybox=False)
    #
    #     proposed_sol_calib_pts.append(control_pts)
    #
    # current_sol_calib_pts = []
    # for each_file in glob.glob('current_sol/*.ply'):
    #     current_sol_cloud = o3d.t.io.read_point_cloud(each_file)
    #     current_sol_cloud = current_sol_cloud.select_by_index(np.where((current_sol_cloud.point.positions[:, 2] > -295) &
    #                                                                    (current_sol_cloud.point.positions[:, 2] < -150))[0])
    #     control_pts = get_control_points(current_sol_cloud, "current_sol")
    #
    #     # Check if control points are found right
    #     current_sol_cloud.paint_uniform_color([0.1, 0.1, 0.1])
    #     control_pts_cloud = o3d.geometry.PointCloud()
    #     control_pts_cloud.points = o3d.utility.Vector3dVector(control_pts)
    #     control_pts_cloud.paint_uniform_color([0, 1, 0])  # green
    #     # See if detected points lie in the right locations
    #     o3d.visualization.draw([{'name': 'pcd', 'geometry': control_pts_cloud, 'material': mat}, current_sol_cloud.to_legacy()],
    #                            show_skybox=False)
    #
    #     current_sol_calib_pts.append(control_pts)

    test_data = []
    for each_file in glob.glob('current_sol/test/*.ply'):
        test_data.append(o3d.t.io.read_point_cloud(each_file))

    # result_calibration = []
    # for i in range(4):
    #     A = current_sol_calib_pts[i].T
    #     B = proposed_sol_calib_pts[i].T
    #     [R, t] = rigid_transform_3D(A, B)
    #     transformed_quadrant = np.dot(R, test_data[i].point.positions.numpy().T) + t
    #     trans_quad_cloud = o3d.geometry.PointCloud()
    #     trans_quad_cloud.points = o3d.utility.Vector3dVector(transformed_quadrant.T)
    #     result_calibration.append(trans_quad_cloud)
    #
    # o3d.visualization.draw([result_calibration[0], result_calibration[1],
    # result_calibration[2], result_calibration[3]])

    proposed_sol_cam1 = o3d.t.io.read_point_cloud("proposed_sol/pyramid_cam1_10000.ply")
    proposed_sol_cam2 = o3d.t.io.read_point_cloud("proposed_sol/pyramid_cam2_10000.ply")
    proposed_sol_cam3 = o3d.t.io.read_point_cloud("proposed_sol/pyramid_cam3_10000.ply")
    proposed_sol_cam4 = o3d.t.io.read_point_cloud("proposed_sol/pyramid_cam4_10000.ply")

    test_cam1 = o3d.t.io.read_point_cloud("current_sol/test/cam1.ply")
    test_cam2 = o3d.t.io.read_point_cloud("current_sol/test/cam2.ply")
    test_cam3 = o3d.t.io.read_point_cloud("current_sol/test/cam3.ply")
    test_cam4 = o3d.t.io.read_point_cloud("current_sol/test/cam4.ply")

    proposed_sol_cam1_control_pts = get_control_points(proposed_sol_cam1, "proposed_sol")
    proposed_sol_cam2_control_pts = get_control_points(proposed_sol_cam2, "proposed_sol")
    proposed_sol_cam3_control_pts = get_control_points(proposed_sol_cam3, "proposed_sol")
    proposed_sol_cam4_control_pts = get_control_points(proposed_sol_cam4, "proposed_sol")

    # The points from one of the data sources is picked manually because with the current calibration object
    # it is not possible to automatically detect which corner is which, as the pyramid is symmetric
    # top, up left, down left, up right, down right
    current_sol_cam1_control_pts = np.asarray([[480.726685, -319.774628, -145.0], [321.4, -144.252563, -305.0],
                                               [307.937622, -482.457947, -305.0], [658.52, -158.633, -305.0],
                                               [647.02, -493.073, -305.0]])
    # top, down left, down right, up left, up right
    current_sol_cam2_control_pts = np.asarray([[-608.974, -256.409, -142.0], [-775.7818, -423.914, -302.0],
                                               [-440.15, -417.354, -302.0], [-777.6168, -88.99, -302.0],
                                               [-443.1768, -85.0523, -302.0]])
    # top, down left, down right, up left, up right
    current_sol_cam3_control_pts = np.asarray([[696.88, 367.6757, -146.0], [525.99, 201.339, -306.0],
                                               [860.12, 200.674, -306.0], [530.2626, 535.0, -306.0],
                                               [862.48, 534.08, -306.0]])
    # top, down right, up right, down left, up left
    current_sol_cam4_control_pts = np.asarray([[-569.858276, 326.09, -146.0], [-406.693359, 163.0572, -306.0],
                                               [-407.7524, 494.226, -306.0], [-741.5439, 164.02, -306.0],
                                               [-737.8175, 497.038, -306.0]])

    [R1, t1] = rigid_transform_3D(current_sol_cam1_control_pts.T, proposed_sol_cam1_control_pts.T)
    [R2, t2] = rigid_transform_3D(current_sol_cam2_control_pts.T, proposed_sol_cam2_control_pts.T)
    [R3, t3] = rigid_transform_3D(current_sol_cam3_control_pts.T, proposed_sol_cam3_control_pts.T)
    [R4, t4] = rigid_transform_3D(current_sol_cam4_control_pts.T, proposed_sol_cam4_control_pts.T)

    transformed_quadrant1 = np.dot(R1, test_cam1.point.positions.numpy().T) + t1
    transformed_quadrant2 = np.dot(R2, test_cam2.point.positions.numpy().T) + t2
    transformed_quadrant3 = np.dot(R3, test_cam3.point.positions.numpy().T) + t3
    transformed_quadrant4 = np.dot(R4, test_cam4.point.positions.numpy().T) + t4

    transformed_quadrant1_cloud = o3d.geometry.PointCloud()
    transformed_quadrant2_cloud = o3d.geometry.PointCloud()
    transformed_quadrant3_cloud = o3d.geometry.PointCloud()
    transformed_quadrant4_cloud = o3d.geometry.PointCloud()

    transformed_quadrant1_cloud.points = o3d.utility.Vector3dVector(transformed_quadrant1.T)
    transformed_quadrant2_cloud.points = o3d.utility.Vector3dVector(transformed_quadrant2.T)
    transformed_quadrant3_cloud.points = o3d.utility.Vector3dVector(transformed_quadrant3.T)
    transformed_quadrant4_cloud.points = o3d.utility.Vector3dVector(transformed_quadrant4.T)

    o3d.visualization.draw_geometries([transformed_quadrant1_cloud, transformed_quadrant2_cloud,
                                       transformed_quadrant3_cloud, transformed_quadrant4_cloud])

    merged_proposed_sol_frame = transformed_quadrant1_cloud + transformed_quadrant2_cloud + \
                                transformed_quadrant3_cloud + transformed_quadrant4_cloud

    merged_current_sol_frame = test_cam1 + test_cam2 + test_cam3 + test_cam4
    o3d.io.write_point_cloud("merged_proposed_sol_frame.pcd", merged_proposed_sol_frame)
    o3d.t.io.write_point_cloud("merged_current_sol_frame.pcd", merged_current_sol_frame)
    o3d.io.write_point_cloud("transformed_quadrant1_cloud.pcd", transformed_quadrant1_cloud)
    o3d.io.write_point_cloud("transformed_quadrant2_cloud.pcd", transformed_quadrant2_cloud)
    o3d.io.write_point_cloud("transformed_quadrant3_cloud.pcd", transformed_quadrant3_cloud)
    o3d.io.write_point_cloud("transformed_quadrant4_cloud.pcd", transformed_quadrant4_cloud)

    print(2)