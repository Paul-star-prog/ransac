# <Your name>
# COMP 776, Fall 2017
# Assignment: RANSAC

import numpy as np

from util import plot_line_inliers

# for purposes of experimentation, we'll keep the sampling fixed every time the
# program is run
np.random.seed(0)


#-------------------------------------------------------------------------------

# Run RANSAC on a dataset for a given model type
#
# @param data                 M x K numpy array containing all M observations in
#                             the dataset, each with K dimensions
# @param inlier_threshold     given some error function for the model (e.g.,
#                             point-to-plane distance if the model is a 3D
#                             plane), label input data as inliers if their error
#                             is smaller than this threshold
# @param confidence_threshold our chosen value p that determines the minimum
#                             confidence required to stop RANSAC
# @param max_num_trials       initial maximum number of RANSAC iterations N
#
# @return best_model          the associated best model for the inliers
# @return inlier_mask         length M numpy boolean array with True indicating
#                             that a data point is an inlier and False
#                             otherwise; inliers can be recovered by taking
#                             data[inlier_mask]
def ransac(data, inlier_threshold, confidence_threshold, max_num_trials):
    max_iter = max_num_trials # current maximum number of trials
    iter_count = 0            # current number of iterations

    best_inlier_count = 0          # initial number of inliers is zero
    best_inlier_mask = np.zeros(   # initially mark all samples as outliers
        len(data), dtype=np.bool)
    best_model = np.array((0., 1., 0.)) # dummy initial model

    # sample size S: two points are sampled for a line model
    S = 2

    # for convenience, put the data into homogeneous coordinates
    data = np.column_stack((data, np.ones(len(data))))


    # continue while the maximum number of iterations hasn't been reached
    while iter_count < max_iter:
        iter_count += 1

        #-----------------------------------------------------------------------
        # 1) sample as many points from the data as are needed to fit the
        #    relevant model

        idxs = np.random.choice(len(data), S, replace=False)
        points2D = data[idxs]


        #-----------------------------------------------------------------------
        # 2) fit a model to the sampled data subset; for a 2D line, the
        #    model parameters should be (n_x, n_y, d) with ||n|| = 1

        model = np.cross(points2D[0], points2D[1])
        model /= np.linalg.norm(model[:2])


        #-----------------------------------------------------------------------
        # 3) determine the inliers to the model; store the result as a boolean
        #    mask, with inliers referenced by data[inlier_mask]

        point_to_line_distance = np.abs(data.dot(model))

        inlier_mask = (point_to_line_distance < inlier_threshold)
        inlier_count = np.count_nonzero(inlier_mask)
        

        #-----------------------------------------------------------------------
        # 4) if this model is the best one yet, update the report and the
        #    maximum iteration threshold

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_ratio = inlier_count / float(len(data))
            best_inlier_mask = inlier_mask
            best_model = model

            # if a perfect model was found, we're done
            if inlier_count == len(data):
                break

            # based on this inlier ratio, re-compute the maximum number of
            # iterations required to achieve the desired probability of a good
            # solution (under the assumption that at least this percentage of
            # samples are inliers)
            #
            # refer to the class slides: slide set 8, slide 71

            # compute (1 - e)^S
            prob_S_samples_are_inliers = np.power(best_inlier_ratio, S)

            # there's a chance that the number of inliers is very small (close
            # to zero); to stably calculate the denominator when solving for N,
            # we'll use the logsumexp trick:
            # log(1 - p) = log(exp(0) - exp(log(p)))
            #            = log(exp(log(p)) * (exp(0)/exp(log(p)) - 1))
            #            = log(p) + log(1 / p - 1)
            denom = (np.log(prob_S_samples_are_inliers) + 
                     np.log(1. / prob_S_samples_are_inliers - 1.))

            if not np.isclose(denom, 0.):
                # solve for the new maximum number of trials N:
                # (1 - (1 - e)^S)^N = 1 - p
                # N * log(1 - (1 - e)^S) = log(1 - p)
                N = np.log(1. - confidence_threshold) / denom

                # use min here because the computed maximum could be greater
                # than max_num_trials
                max_iter = min(max_iter, N)


    #---------------------------------------------------------------------------
    # print the best models found before refinement

    print "Inlier Ratio (before refinement): {:.3f}".format(best_inlier_ratio)
    print "Best Fit Model (before refinement): {:.5f}, {:.5f}, {:.5f}".format(
        *best_model)
    print


    #---------------------------------------------------------------------------
    # 5) run a final least-squares fit on the line equation using the inliers
    #    Here, we use the DLT: ax + by + c = 0 for all our points, so we can
    #    build a data matrix A with rows [ x_i y_i 1] and solve for
    #    A . [a b c]^T = 0

    A = data[best_inlier_mask]
    _, _, VT = np.linalg.svd(A)
    model = VT[-1] # solution is the row of V^T corresponding to the smallest
                   # eigenvalue
    model /= np.linalg.norm(model[:2]) # normalize the line equation

    point_to_line_distance = np.abs(data.dot(model))
    inlier_mask = (point_to_line_distance < inlier_threshold)
    inlier_ratio = np.count_nonzero(inlier_mask) / float(len(data))


    #--------------------------------------------------------------------------
    # print some information about the results of RANSAC

    print "Iterations:", iter_count
    print "Inlier Ratio: {:.3f}".format(inlier_ratio)
    print "Best Fit Model: {:.5f}, {:.5f}, {:.5f}".format(*model)

    return model, inlier_mask


#-------------------------------------------------------------------------------

# program main
# @param args command-line arguments
def main(args):
    # generate a random line perturbed with Gaussian noise of sigma=0.1
    M = 100
    xmin, xmax = -10., 10.
    x = np.random.rand(M) * (xmax - xmin) + xmin
    y = 0.1 * np.random.randn(M)
    data = np.column_stack((x, y)) # Mx2

    # to make things more interesting, rotate and translate the data
    theta = np.radians(25.)
    R = np.array(((np.cos(theta), -np.sin(theta)),
                  (np.sin(theta),  np.cos(theta))))
    t = np.array((-0.5, 1.5))

    data = data.dot(R.T)
    data += t

    # for illustration purposes, print the parameters for the line we've
    # generated
    n_x = np.sin(theta)
    n_y = -np.cos(theta)
    d = -t.dot((n_x, n_y))

    print "Actual Model: {:.5f}, {:.5f}, {:.5f}".format(n_x, n_y, d)
    print


    #---------------------------------------------------------------------------
    # add random outliers in the range [-50,50]x[-50,50]

    M_outliers = 50
    outliers = 10. * np.column_stack(
        (2. * np.random.rand(M_outliers) - 1.,
         2. * np.random.rand(M_outliers) - 1.))

    data = np.row_stack((data, outliers))


    #---------------------------------------------------------------------------
    # run RANSAC

    model, inlier_mask = ransac(data, args.inlier_threshold,
                                args.confidence_threshold, args.max_num_trials)

    # plot the results
    plot_line_inliers(data, inlier_mask)
        

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit a 2D line model using RANSAC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #
    # RANSAC options
    #

    parser.add_argument("--inlier_threshold", type=float, default=0.3,
        help="point-to-line distance threshold to use for RANSAC")

    parser.add_argument("--confidence_threshold", type=float, default=0.99,
        help="stop RANSAC when the probability that a correct model has been "
             "found reaches this threshold")

    parser.add_argument("--max_num_trials", type=float, default=10000,
        help="maximum number of RANSAC iterations to allow")

    args = parser.parse_args()

    main(args)
