# <Your name>
# COMP 776, Fall 2017
# Assignment: RANSAC

import numpy as np

import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
# plots the inliers for a line fit to 2D data
#
# @param data             M x 3 matrix of 3D coordinates
# @param inlier_mask      M-length boolean vector with inliers denoted as True
def plot_line_inliers(data, inlier_mask):
    # create a plot
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # plot the inliers in blue and outliers in red
    ax.scatter(data[inlier_mask,0], data[inlier_mask,1], c="b")
    ax.scatter(data[~inlier_mask,0], data[~inlier_mask,1], c="r")

    # show the plot
    plt.show()


#-------------------------------------------------------------------------------
# plots the epipolar lines for a two images related by a fundamental matrix
#
# @param image1_path       path to first image
# @param image2_path       path to second image
# @param keypoints1        M x 2 matrix of (x,y) coordinates in the first image
# @param keypoints2        M x 2 matrix of (x,y) coordinates in the second image
# @param F                 fundamental matrix relating the two images
# @param inlier_mask       M-length boolean vector with inliers denoted as True
def plot_epipolar_inliers(image1_path, image2_path, keypoints1, keypoints2, F,
                          inlier_mask):

    image1 = plt.imread(image1_path)
    image2 = plt.imread(image2_path)
    both_images = np.column_stack((image1, image2))

    # for convenience, make keypoints homogeneous
    keypoints1 = np.column_stack((keypoints1, np.ones(len(keypoints1))))
    keypoints2 = np.column_stack((keypoints2, np.ones(len(keypoints2))))

    # inlier_mask = np.random.choice(inlier_mask, 20)

    image1_inliers = keypoints1[inlier_mask, :]
    image2_inliers = keypoints2[inlier_mask, :]

    image1_outliers = keypoints1[~inlier_mask, :]
    image2_outliers = keypoints2[~inlier_mask, :]

    offset = image1.shape[1] # x offset when concatenating the second image

    #--------------------------------------------------------------------------
    # plot inliers

    plt.title("Inlier Correspondences")
    plt.imshow(both_images)
    plt.scatter(keypoints1[:,0], keypoints1[:,1], c='r', s=4)
    plt.scatter(keypoints2[:,0] + offset, keypoints2[:,1], c='b', s=4)

    for i in xrange(image1_inliers.shape[0]):
        plt.plot(
            (image1_inliers[i,0], image2_inliers[i,0] + offset),
            (image1_inliers[i,1], image2_inliers[i,1]),
            linewidth=1.0, c='g')
    plt.savefig('InlierCorresp.png')
    
    #--------------------------------------------------------------------------
    # plot epipolar lines for the inliers

    # compute the intersection of the epipolar lines with the image boundary
    def compute_endpoints(epi_lines, max_x, max_y):
        y_left = -epi_lines1[:,2] / epi_lines1[:,1]
        y_right = -(epi_lines[:,0] * max_x + epi_lines1[:,2]) / epi_lines1[:,1]
        x_top = -epi_lines1[:,2] / epi_lines1[:,0]
        x_bottom = -(epi_lines[:,1] * max_y + epi_lines1[:,2]) / epi_lines1[:,0]

        left_mask = (y_left >= 0.) & (y_left <= max_y)
        right_mask = (y_right >= 0.) & (y_right <= max_y)
        top_mask = (x_top >= 0.) & (x_top <= max_x)
        bottom_mask = (x_bottom >= 0.) & (x_bottom <= max_x)

        # for the first point, select the left intersection (if valid), top
        # intersection (if valid), or bottom intersection (otherwise)
        # for the second point, prefer right, bottom, or top

        x1 = np.zeros(len(epi_lines))
        y1 = y_left
        x2 = max_x + np.zeros(len(epi_lines))
        y2 = y_right

        mask = (~left_mask) & top_mask
        x1[mask] = x_top[mask]
        y1[mask] = 0.

        mask = (~left_mask) & (~top_mask)
        x1[mask] = x_bottom[mask]
        y1[mask] = max_y

        mask = (~right_mask) & bottom_mask
        x2[mask] = x_bottom[mask]
        y2[mask] = 0.

        mask = (~right_mask) & (~bottom_mask)
        x2[mask] = x_top[mask]
        y2[mask] = max_y

        return x1, y1, x2, y2

    epi_lines1 = image2_inliers.dot(F) # lines in first image
    epi_lines2 = image1_inliers.dot(F.T) # lines in second image

    # to avoid divisions by zero, we'll set zeros to a small positive constant
    epi_lines1[np.isclose(epi_lines1, 0.)] = np.finfo("float").eps
    epi_lines2[np.isclose(epi_lines2, 0.)] = np.finfo("float").eps

    x11, y11, x12, y12 = compute_endpoints(
        epi_lines1, image1.shape[1] - 1, image1.shape[0])
    x21, y21, x22, y22 = compute_endpoints(
        epi_lines2, image2.shape[1] - 1, image2.shape[0])

    print keypoints1[:,0]
    print keypoints2[:,0]

    plt.figure()
    plt.title("Inlier Epipolar Lines")
    plt.imshow(both_images)
    plt.scatter(keypoints1[:,0], keypoints1[:,1], c='r', s=4)
    plt.scatter(keypoints2[:,0] + offset, keypoints2[:,1], c='b', s=4)

    for i in xrange(image1_inliers.shape[0]):
        plt.plot((x11[i], x12[i]), (y11[i], y12[i]), linewidth=1.0, c='g')
        plt.plot((x21[i] + offset, x22[i] + offset), (y21[i], y22[i]),
                 linewidth=1.0, c='g')
    plt.savefig('InlierEpipolarLines.png')

    
    #--------------------------------------------------------------------------
    # plot outliers

    # plot the outlier points, images, and matches
    plt.figure()
    plt.title("Outlier Correspondences")
    plt.imshow(both_images)
    plt.scatter(keypoints1[:,0], keypoints1[:,1], c='r', s=4)
    plt.scatter(keypoints2[:,0] + offset, keypoints2[:,1], c='b', s=4)

    for i in xrange(image1_outliers.shape[0]):
        plt.plot(
            (image1_outliers[i,0], image2_outliers[i,0] + offset),
            (image1_outliers[i,1], image2_outliers[i,1]),
            linewidth=1.0, c='y')

    # plt.show()
    plt.savefig('OutlierCorresp.png')
