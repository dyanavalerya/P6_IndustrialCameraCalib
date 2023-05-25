import open3d as o3d
import numpy as np

import argparse
import time
import matplotlib.pyplot as plt
from random import sample

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_mask(ind)
    outlier_cloud = cloud.select_by_mask(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0]) # red
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    # o3d.visualization.draw_geometries([inlier_cloud.to_legacy(), outlier_cloud.to_legacy()])
    o3d.visualization.draw_geometries([inlier_cloud.to_legacy()])

# Remember to change camera number depending on data from the arguments
if __name__ == '__main__':
    pcd_data = o3d.data.DemoICPPointClouds()
    cam_nr = 1
    parser = argparse.ArgumentParser(
        'Noise determination on the Z axis using cell plane fitting')
    parser.add_argument('cloud',
                        type=str,
                        default=pcd_data.paths[0],
                        nargs='?',
                        help='path to measurement point cloud')

    args = parser.parse_args()

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    print('Reading inputs')
    cloud = o3d.t.io.read_point_cloud(args.cloud)

    # o3d.visualization.draw_geometries([cloud.to_legacy()])
    # Step 1: Threshold membrane and noise and leave only cloud of machined object
    # Use z axis positions as limit values (I used Cloud Compare, and looked at Scalar field for different points)
    cloud = cloud.select_by_index(np.where((cloud.point.positions[:, 2] > -250) & (cloud.point.positions[:, 2] < 0))[0])

    # Step 2: make cloud straight at the edges by removing unwanted points
    # If for some reason noise from the membrane was caught up at the same z level as the machined object
    intensity = cloud.point.scalar_Scalar_field.numpy().flatten()
    if cam_nr == 2 | cam_nr == 4:
        cloud = cloud.select_by_mask(intensity > 60.0)
    else:
        cloud = cloud.select_by_mask(intensity > 50.0)
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

    # Conversion to legacy point cloud to be able to use the voxelization methods
    cloud = cloud.to_legacy()

    # Voxelization will be used to split the cloud into a grid and later
    # be able to collect points corresponding to each segment
    # The voxel size is chosen dependent on the cloud that is being passed as input
    # and the size of the square segments, here I try to aim at 50 x 50 mm cells
    # For a cloud of size 2120 x 1400 mm (the size of the unaltered one) we get 1176 cells, so trial and error
    # until approximately 1176 are achieved
    # For a cloud of size 910 x 550 mm (the size of an altered quadrant) we get 200 cells
    print('Voxelization')

    # Time voxelization code block for testing performance
    t0 = time.time()
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cloud, voxel_size=60)
    voxels = voxel_grid.get_voxels()
    # Sample through the voxels list in order to shrink it substantially
    # voxels = sample(voxels, 20)

    # o3d.visualization.draw_geometries([voxel_grid])

    queries = np.asarray(cloud.points)

    # make a new array that only stored the id of the voxels and not all the other
    # irrelevant information fot the intended purposes
    voxel_index = np.zeros(shape=(len(voxels), 3), dtype=np.int32)
    for k in range(len(voxels)):
        voxel_index[k] = voxels[k].grid_index

    # initialize a list to store the list of point lists later used to fit a plane onto each cell
    # to check for the noise level onto small areas of the cloud
    # the plane fitting is done in matlab file fit_planes_on_grid_cloud.mlx
    split_points = list()
    for i in range(0, len(voxels)):
        split_points.append(list())

    print("Get the list of points arranged per voxel")
    # Loop through all points of the cloud yet to be segmented
    for i in range(len(queries)):
        # get the id of the voxel that contains a point queries[i]
        voxel_id = voxel_grid.get_voxel(queries[i])
        for k in range(len(voxels)):
            # find the location of that voxel in the list and save the cloud point
            # in that same location in order to create list of lists
            # with each individual list being the points under a voxel, i.e. being part of a cell
            # with the amount of point list the same as the number of voxels
            if (voxel_index[k, :] == voxel_id).all():
                # save point in list
                split_points[k].append(queries[i])
                break

    t1 = time.time()
    total_time = t1 - t0
    print("Time it took to segment the cloud: ", total_time)

    #np.save('cloud_grid_50_x_50_straight400_cloud_sampled', np.asarray(split_points))

    # Access each point list
    # I save in csv format because that is one file format I can open in Matlab
    # I also save each list in separate files for ease of access in Matlab
    # and because it was problematic to save a list of lists in csv format
    for i, list in enumerate(split_points):
        if list != []:
            np.savetxt("cloud_grid_50_x_50_cam" + str(cam_nr) + "_m1" + str(i) + ".csv", np.asarray(list), delimiter=",")
