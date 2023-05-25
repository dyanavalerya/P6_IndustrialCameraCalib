import open3d as o3d
import numpy as np

import argparse
import matplotlib.pyplot as plt

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0]) # red
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    o3d.visualization.draw_geometries([inlier_cloud])

if __name__ == '__main__':
    pcd_data = o3d.data.DemoICPPointClouds()
    parser = argparse.ArgumentParser(
        'Global point cloud registration example with RANSAC')
    parser.add_argument('src',
                        type=str,
                        default=pcd_data.paths[0],
                        nargs='?',
                        help='path to src point cloud')
    parser.add_argument('dst',
                        type=str,
                        default=pcd_data.paths[1],
                        nargs='?',
                        help='path to dst point cloud')

    args = parser.parse_args()

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    print('Reading inputs')
    dst = o3d.io.read_point_cloud(args.dst)
    src = o3d.io.read_point_cloud(args.src)

    # o3d.visualization.draw_geometries([src, dst])

    print("Statistical outlier removal")  # (do this only when using many points as source input)

    # The lower the standard deviation is the more aggressive the filter will be
    # The more neighbours you put the higher change that the outliers will be removed,
    # but also if you set it too high too many points from the margins will be removed
    # use nb_neighbours = 15 and std_ratio = 0.5 for the down sampled cloud
    # use nb_neighbours = 200 and std_ratio = 0.3 for the not down sampled cloud
    cl, ind = src.remove_statistical_outlier(nb_neighbors=200,
                                                  std_ratio=0.3)

    cl2, ind2 = dst.remove_statistical_outlier(nb_neighbors=200,
                                             std_ratio=0.3)

    # display_inlier_outlier(src, ind)
    # display_inlier_outlier(dst, ind2)
    # Keep only the inliers (do this only when using many points as source input,
    # but keep in mind that ICP would take ages)
    #
    # The outlier removal does NOT shuffle the position of the remaining points
    # Thus they remain in the original location captured by the sensor
    src = src.select_by_index(ind)
    dst = dst.select_by_index(ind2)

    # Move dst slightly on the z axis
    # np.asarray(dst.points)[:, 2] = np.asarray(dst.points)[:, 2] + 0.5

    print("Calculate distance error metrics")
    # Calculate absolute distances using flann
    distances = o3d.geometry.PointCloud.compute_point_cloud_distance(src, dst)

    dst_tree = o3d.geometry.KDTreeFlann(dst) # change here for another target
    # [k, idx, _] = dst_tree.search_knn_vector_3d(src.points[1512], 1)
    dst_knn_arranged = []
    distances2 = []
    for i, point in enumerate(src.points):
        [k, idx, _] = dst_tree.search_knn_vector_3d(src.points[i], 1)
        dst_knn_arranged.append(np.asarray(dst.points)[idx][0]) # change here for another target
        distances2.append(_[0])
    # np.asarray(src.colors)[1512] = [1, 1, 0]
    # np.asarray(dst.colors)[idx] = [1, 0, 1] # change here for another target

    # Calculate distance on z axis between target and source point
    distances_z = np.subtract(np.asarray(dst_knn_arranged)[:, 2], np.asarray(src.points)[:, 2])

    # Calculate deviations by performing the dot product of the normal to the vector between the
    # target and source points AB to find its orientation (in this case above or below the surface)
    # and multiply it with the magnitude of AB
    # AxBx_vectors = np.subtract(np.asarray(dst_knn_arranged), np.asarray(src.points))
    # deviation = []
    # for i in range(np.asarray(src.points).shape[0]):
    #     # Get the norm (magnitude) of vector AB
    #     magnitudeAB = dist(np.asarray(src.points[i]), np.asarray(dst_knn_arranged[i]))
    #     # Get the projection as result of the normal perpendicular to AB
    #     projectionNtoAB = np.dot(np.asarray(dst.normals[i]), AxBx_vectors[i])
    #     deviation.append(np.sign(projectionNtoAB) * np.asarray(magnitudeAB))

    src.paint_uniform_color([1, 0, 0]) # red
    dst.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([src, dst])

    print("Visualize point pairs after kd-tree knn search")
    # Show point pairs (visualize vectors)

    points = []
    for i in range(np.asarray(src.points).shape[0]):
        points.append(np.asarray(src.points[i]))
        points.append(np.asarray(dst_knn_arranged[i]))

    lines = []
    for i in range(2 * np.asarray(src.points).shape[0]):
        if i % 2 == 0:
            lines.append([i, i + 1])

    colors = [[0, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([line_set, src, dst])

    print("Plot results")

    # plt.figure(1)
    # plt.title("Distance errors")
    # plt.xlabel("Bins")
    # plt.ylabel("Nr. of occurrences")
    # plt.bar(list(range(1, len(distances) + 1)), distances, width=1)
    #plt.hist(deviation, 50)
    # plt.scatter(list(range(1, len(distances) + 1)), distances, c="black")

    # plt.hist(src_fpfh.data.ravel()) # test to plot fpfh data and see how it looks (not directly related to the above plots)
    # plt.show()

    # The closer to zero the distance the better fit it has
    # opposite for the matplotlib colormaps: the lighter color the smaller the distance, and more hue the bigger dist.
    # Greens = mpl.colormaps['Greens'].resampled(256)
    # plot_clm([Greens], np.reshape(distances, (-1, len(distances))))
    # for some reason does not contain the color of each point, i.e. array not the same size as nr. of pts.
    # colors_arr = Greens._lut

    # The darker the color the smaller the distance, the more hue the bigger the distance (my solution)
    # The smaller the color value the darker the color
    # plt.figure(1)
    # plt.title("Absolute distance plot")
    # plt.bar(list(range(1, len(distances) + 1)), distances, width=1)
    # plt.show()

    # Plot distances on z
    #plt.figure(2)
    # plt.bar(list(range(1, len(distances_z) + 1)), distances_z, width=1)
    # plt.title("Distances on Z axis between src and dst")
    # plt.show()

    # Plot deviations
    # plt.figure(4)
    # plt.title("Deviations plot")
    # plt.bar(list(range(1, len(deviation) + 1)), deviation, width=1)
    # plt.show()

    # Create color map for the source cloud (the darker the colors the smaller the distance and the better)
    # Green means the distance is within (-1, 1) mm
    # colors = []
    # for d in deviation:
    #     colors.append(get_color_array2(np.asarray(deviation), d))

    # colors_down = get_color_array(distances, "green")

    # src.colors = o3d.utility.Vector3dVector(colors)
    # src_down.colors = o3d.utility.Vector3dVector(colors_down)

    # o3d.visualization.draw([src], point_size=30)