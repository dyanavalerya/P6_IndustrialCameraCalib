import open3d as o3d
import numpy as np

import time
import glob
import matplotlib.pyplot as plt
from math import cos, sin, pi

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_mask(ind)
    outlier_cloud = cloud.select_by_mask(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0]) # red
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud.to_legacy(), outlier_cloud.to_legacy()])
    o3d.visualization.draw_geometries([inlier_cloud.to_legacy()])

def least_squares_line(edge_array, name, measurement_nr, cam_nr):
    # y = np.asarray(list)[:, 1]
    # x = np.asarray(list)[:, 0]
    y = edge_array[:, 1]
    x = edge_array[:, 0]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    _ = plt.plot(x, y, 'o', label='Original data', markersize=5)
    _ = plt.plot(x, m * x + c, 'r', label='Fitted line')
    if name == 'left_edge':
        _ = plt.title('Camera ' + str(cam_nr) + ': Measurement ' + str(measurement_nr) + ' left edge')
        _ = plt.savefig('Camera' + str(cam_nr) + '_measurement' + str(measurement_nr) + '_left_edge.png', bbox_inches='tight')
    elif name == 'right_edge':
        _ = plt.title('Camera ' + str(cam_nr) + ': Measurement ' + str(measurement_nr) + ' right edge')
        _ = plt.savefig('Camera' + str(cam_nr) + '_measurement' + str(measurement_nr) + '_right_edge.png', bbox_inches='tight')
    elif name == 'upper_edge':
        _ = plt.title('Camera ' + str(cam_nr) + ': Measurement ' + str(measurement_nr) + ' upper edge')
        _ = plt.savefig('Camera' + str(cam_nr) + '_measurement' + str(measurement_nr) + '_upper_edge.png', bbox_inches='tight')
    else:
        _ = plt.title('Camera ' + str(cam_nr) + ': Measurement ' + str(measurement_nr) + ' lower edge')
        _ = plt.savefig('Camera' + str(cam_nr) + '_measurement' + str(measurement_nr) + '_lower_edge.png', bbox_inches='tight')
    _ = plt.legend()
    _ = plt.show()
    return m, c

def threshold_membrane(cloud):
    labels = np.zeros(shape=(len(cloud.point.positions)), dtype=bool)
    for i, pt in enumerate(cloud.point.positions):
        if pt[2] > -250:
            labels[i] = 1
        else:
            labels[i] = 0
    return labels

def estimate_boundary(cloud, cam_nr):
    boundaries, mask = cloud.compute_boundary_points(3, 16)
    print(f"Detect {boundaries.point.positions.shape[0]} boundary points from {cloud.point.positions.shape[0]} points.")

    boundaries = boundaries.paint_uniform_color([1.0, 0.0, 0.0])
    cloud = cloud.paint_uniform_color([0.6, 0.6, 0.6])

    # Remove all points inside the border rectangle, as they are considered noise
    # The noise is a consequence of not so good boundary detection parameters
    # The limits for x and y were achieved by analyzing the boundary plot for each quadrant
    remove = np.zeros(shape=(len(boundaries.point.positions)), dtype=bool)
    for i, pt in enumerate(boundaries.point.positions):
        if cam_nr == 1:
            if (pt[0] > 150.0) & (pt[0] < 950.0) & (pt[1] > -550.0) & (pt[1] < -150.0):
                remove[i] = 0
            else:
                remove[i] = 1
        elif cam_nr == 2:
            if (pt[0] > -915.0) & (pt[0] < -140.0) & (pt[1] > -485.0) & (pt[1] < -110.0):
                remove[i] = 0
            else:
                remove[i] = 1
        elif cam_nr == 3:
            if (pt[0] > 200.0) & (pt[0] < 982.0) & (pt[1] > 110.0) & (pt[1] < 510.0):
                remove[i] = 0
            else:
                remove[i] = 1
        elif cam_nr == 4:
            if (pt[0] > -940.0) & (pt[0] < -140.0) & (pt[1] > 125.0) & (pt[1] < 535.0):
                remove[i] = 0
            else:
                remove[i] = 1

    boundaries = boundaries.select_by_mask(remove)
    # o3d.visualization.draw_geometries([boundaries.to_legacy()],
    #                                   zoom=0.3412,
    #                                   front=[0.3257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])

    # The rest of the function commands are used for debugging and code development

    # y = boundaries.point.positions.numpy()[:, 1]
    # x = boundaries.point.positions.numpy()[:, 0]
    #
    # _ = plt.plot(x, y, 'x', label='Cloud boundary', markersize=3)
    # _ = plt.legend()
    # _ = plt.show()

    # Save boundary data to load in Matlab for analysis
    # np.savetxt("boundary_cam4.csv", boundaries.point.positions.numpy(), delimiter=",")

    return boundaries

def rotate3d(theta):
    return np.asarray([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])

def rotate2d(theta):
    return np.asarray([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

def edge_selection(boundaries_cloud, cam_nr):
    # I choose to work with lists because I don't know how large they will be
    left_edge = []
    right_edge = []
    upper_edge = []
    lower_edge = []
    # Analyze plot of boundary and set the x and y limits for each of the edges
    # I saved the boundary as a csv file, and loaded in Matlab as it is much easier
    # to look at the points and their exact coordinates
    for i, pt in enumerate(boundaries_cloud.point.positions.numpy()):
        if cam_nr == 1:
            # left edge
            if (pt[0] > 0) & (pt[0] < 140):
                left_edge.append(pt)
            # right edge
            if (pt[0] > 955) & (pt[0] < 1000):
                right_edge.append(pt)
            # upper edge
            if (pt[1] > -140) & (pt[1] < 0):
                upper_edge.append(pt)
            # lower edge
            if (pt[1] > -600) & (pt[1] < -560):
                lower_edge.append(pt)
        elif cam_nr == 2:
            # left edge #922
            if (pt[0] > -1000) & (pt[0] < -930):
                left_edge.append(pt)
            # right edge
            if (pt[0] > -127) & (pt[0] < -100):
                right_edge.append(pt)
            # upper edge
            if (pt[1] > -107) & (pt[1] < -50):
                upper_edge.append(pt)
            # lower edge
            if (pt[1] > -520) & (pt[1] < -495):
                lower_edge.append(pt)
        elif cam_nr == 3:
            # left edge
            if (pt[0] > -110) & (pt[0] < 175):
                left_edge.append(pt)
            # right edge
            if (pt[0] > 980) & (pt[0] < 1005):
                right_edge.append(pt)
            # upper edge
            if (pt[1] > 515) & (pt[1] < 560):
                upper_edge.append(pt)
            # lower edge
            if (pt[1] > 70) & (pt[1] < 108):
                lower_edge.append(pt)
        elif cam_nr == 4:
            # left edge # <955
            if (pt[0] > -1000) & (pt[0] < -954):
                left_edge.append(pt)
            # right edge
            if (pt[0] > -135) & (pt[0] < -125):
                right_edge.append(pt)
            # upper edge
            if (pt[1] > 548) & (pt[1] < 600):
                upper_edge.append(pt)
            # lower edge
            if (pt[1] > 114) & (pt[1] < 120):
                lower_edge.append(pt)
    # rotate all edges by 45 degrees to avoid having vertical axes with infinite slope
    left_edge = np.transpose(np.dot(rotate3d(pi/4), np.transpose(np.asarray(left_edge))))
    right_edge = np.transpose(np.dot(rotate3d(pi / 4), np.transpose(np.asarray(right_edge))))
    upper_edge = np.transpose(np.dot(rotate3d(pi / 4), np.transpose(np.asarray(upper_edge))))
    lower_edge = np.transpose(np.dot(rotate3d(pi / 4), np.transpose(np.asarray(lower_edge))))

    return left_edge, right_edge, upper_edge, lower_edge

def find_intersection_pts(left_edge, right_edge, upper_edge, lower_edge, boundaries_cloud, measurement_nr, cam_nr):
    # Fit lines through point cloud edges
    m1, c1 = least_squares_line(left_edge, 'left_edge', measurement_nr, cam_nr)
    m2, c2 = least_squares_line(right_edge, 'right_edge', measurement_nr, cam_nr)
    m3, c3 = least_squares_line(upper_edge, 'upper_edge', measurement_nr, cam_nr)
    m4, c4 = least_squares_line(lower_edge, 'lower_edge', measurement_nr, cam_nr)

    # and their intersection points
    # x = (b2 - b1)/(m1-m2) from y1 = y2 = m1x+b1 = m2x + b2
    # y = m1*x+c1
    up_left_corner = [(c3-c1)/(m1-m3), m1*((c3-c1)/(m1-m3))+c1]
    down_left_corner = [(c4-c1)/(m1-m4), m1*((c4-c1)/(m1-m4))+c1]
    up_right_corner = [(c3-c2)/(m2-m3), m2*((c3-c2)/(m2-m3))+c2]
    down_right_corner = [(c4-c2)/(m2-m4), m2*((c4-c2)/(m2-m4))+c2]

    # Rotate corners back by -45 deg to make lines vertical and horizontal
    up_left_corner = np.dot(rotate2d(-pi/4), np.transpose(np.asarray(up_left_corner)))
    down_left_corner = np.dot(rotate2d(-pi / 4), np.transpose(np.asarray(down_left_corner)))
    up_right_corner = np.dot(rotate2d(-pi / 4), np.transpose(np.asarray(up_right_corner)))
    down_right_corner = np.dot(rotate2d(-pi / 4), np.transpose(np.asarray(down_right_corner)))

    # Plot boundary cloud for visual testing purposes together with the calculated corners
    corners = [up_left_corner, down_left_corner, up_right_corner, down_right_corner]
    x_corners = np.asarray(corners)[:, 0]
    y_corners = np.asarray(corners)[:, 1]

    y = boundaries_cloud.point.positions.numpy()[:, 1]
    x = boundaries_cloud.point.positions.numpy()[:, 0]

    _ = plt.plot(x_corners, y_corners, 'o', label='Intersection corners', markersize=10)
    _ = plt.plot(x, y, 'x', label='Cloud boundary', markersize=3)
    _ = plt.title('Camera ' + str(cam_nr) + ': Measurement ' + str(measurement_nr))
    _ = plt.legend()
    _ = plt.savefig('Camera' + str(cam_nr) + '_measurement'+str(measurement_nr)+'_corners.png', bbox_inches='tight')
    _ = plt.show()

    return corners

def x_y_noise(cloud, measurement_nr, cam_nr):
    # o3d.visualization.draw_geometries([cloud.to_legacy()])
    print("Selecting data from measurement " + str(measurement_nr))
    # Step 1: Threshold membrane and noise and leave only cloud of machined object
    # Use z axis positions as limit values (I used Cloud Compare, and looked at Scalar field for different points)
    # labels = threshold_membrane(cloud)
    cloud = cloud.select_by_index(np.where((cloud.point.positions[:, 2] > -250) & (cloud.point.positions[:, 2] < 0))[0])

    # Step 2: make cloud straight at the edges by removing unwanted points
    # If for some reason noise from the membrane was caught up at the same z level as the machined object
    intensity = cloud.point.scalar_Scalar_field.numpy().flatten()
    if cam_nr == 2 | cam_nr == 4:
        cloud = cloud.select_by_mask(intensity > 60.0)
    else:
        cloud = cloud.select_by_mask(intensity > 50.0)

    # cloud = cloud.select_by_mask(labels & (intensity > 50.0))
    # o3d.visualization.draw_geometries([cloud.to_legacy()])

    # Remove rest of points that seem to make edges not straight
    # The parameters were achieved by trial and error
    if cam_nr == 1:
        # Camera 1 16 4
        cl, ind = cloud.remove_radius_outliers(nb_points=16, search_radius=4)
    elif cam_nr == 2:
        cl, ind = cloud.remove_radius_outliers(nb_points=20, search_radius=4)
    elif cam_nr == 3:
        # Camera 3 16 4 or 14 5
        cl, ind = cloud.remove_radius_outliers(nb_points=14, search_radius=5)
    elif cam_nr == 4:
        cl, ind = cloud.remove_radius_outliers(nb_points=30, search_radius=4)
    # display_inlier_outlier(cloud, ind)

    # Keep only the inliers
    cloud = cloud.select_by_mask(ind)

    # Step 3: Find the boundary points
    boundaries = estimate_boundary(cloud, cam_nr)
    # o3d.visualization.draw_geometries([boundaries.to_legacy()])

    # Step 4: Select each of the edges separately
    left_edge, right_edge, upper_edge, lower_edge = edge_selection(boundaries, cam_nr)

    # Step 5: Find intersection corners by fitting lines to the edges using least-squares
    corners = find_intersection_pts(left_edge, right_edge, upper_edge, lower_edge, boundaries, measurement_nr, cam_nr)
    return corners


# Change camera_nr to switch between data files
if __name__ == '__main__':
    t0 = time.time()
    print('Reading inputs')
    # Change the path to the different measurements
    measurements = []
    camera_nr = 1
    for each_file in glob.glob('calib/x_y_noise_data/cam' + str(camera_nr) + '/*.ply'):
        measurements.append(o3d.t.io.read_point_cloud(each_file))

    corners_per_measurement = []
    for i, measurement in enumerate(measurements):
        # Make sure to change the camera number depending on the data being passed
        corners_per_measurement.append(x_y_noise(measurement, measurement_nr=i+1, cam_nr=camera_nr))


    up_left_corner = np.asarray(corners_per_measurement)[:, 0]
    down_left_corner = np.asarray(corners_per_measurement)[:, 1]
    up_right_corner = np.asarray(corners_per_measurement)[:, 2]
    down_right_corner = np.asarray(corners_per_measurement)[:, 3]

    np.savetxt("up_left_corner_data_cam" + str(camera_nr) + ".csv", up_left_corner, delimiter=",")
    np.savetxt("down_left_corner_data_cam" + str(camera_nr) + ".csv", down_left_corner, delimiter=",")
    np.savetxt("up_right_corner_data_cam" + str(camera_nr) + ".csv", up_right_corner, delimiter=",")
    np.savetxt("down_right_corner_data_cam" + str(camera_nr) + ".csv", down_right_corner, delimiter=",")

    # Calculate variance for each corner point difference between measurements
    # resulting in four variance values per quadrant

    variance = [np.var(up_left_corner, axis=0), np.var(down_left_corner, axis=0),
                np.var(up_right_corner, axis=0), np.var(down_right_corner, axis=0)]

    np.savetxt("variance_cam" + str(camera_nr) + ".csv", np.asarray(variance), delimiter=",")

    t1 = time.time()
    total_time = t1 - t0
    print("Time it took to find corners of " + str(len(measurements)) + " measurements: ", total_time)

    print(2)