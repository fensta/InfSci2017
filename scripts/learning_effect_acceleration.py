"""
Here we analyze when the speed of the learning curve declines. To do so, after
fitting a polynomial to the first N tweets, we compute its 2nd derivative and
check where its slope flattens out.
IMPORTANT: all labels are used, i.e. if "irrelevant" was chosen by annotator,
the other assigned labels to that tweet are also used in order to reduce the
asymmetry in our hierarchical annotation scheme.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly

import pseudo_db


# For Information sciences
FONTSIZE = 12
plt.rcParams.update({'font.size': FONTSIZE})


def plot_learning_acceleration(dst, n, to_keep, md_tw_p, md_a_p, su_tw_p,
                               su_a_p, cleaned=False):
    """
    Plots how the acceleration of the learning effect develops at different
    intervals.

    Parameters
    ----------
    dst: str - directory in which the plot will be stored.
    n: int - position at which the fit of the first polynomial should end latest
    and the fit of the 2nd polynomial for the remaining data points should start
    to_keep: int - number of tweets to keep per annotator for fitting.
    md_tw_p: str - path to MD tweets dataset in csv format.
    md_a_p: str - path to MD annotators dataset in csv format.
    su_tw_p: str - path to SU tweets dataset in csv format.
    su_a_p: str - path to SU annotators dataset in csv format.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing average
    annotation times.

    Raises
    ------
    ValueError: if to_keep is not 3 or more larger than n, which is necessary to
    fit the 2nd polynomial of degree 3.

    """
    # if to_keep - n < 3:
    #     raise ValueError("Need at least 3 points for fitting the 2nd polynomial"
    #                      "")

    dataset_type = "raw"
    if cleaned:
        dataset_type = "cleaned"

    # Each inner list represents the first <to_keep> annotation times of a
    # single annotator
    # {institution: {group: [[1, 3], [4, 6, 7, 2]]}}
    times = {
        "md": get_total_anno_times(md_tw_p, md_a_p, to_keep, cleaned=cleaned),
        "su": get_total_anno_times(su_tw_p, su_a_p, to_keep, cleaned=cleaned)
    }
    print "MD:", times["md"]
    print "SU:", times["su"]

    ###################################
    # Compute medians per institution #
    ###################################
    # All annotator groups
    inst_times_md = []
    inst_times_su = []
    # Only S, M annotator groups
    inst_times_md_sm = []
    inst_times_su_sm = []
    for inst in times:
        if inst == "md":
            inst_times_md.extend(times[inst]["S"])
            inst_times_md_sm.extend(times[inst]["S"])
            if "M" in times[inst]:
                inst_times_md.extend(times[inst]["M"])
                inst_times_md_sm.extend(times[inst]["M"])
            if "L" in times[inst]:
                inst_times_md.extend(times[inst]["L"])
        else:
            inst_times_su.extend(times[inst]["S"])
            inst_times_su_sm.extend(times[inst]["S"])
            if "M" in times[inst]:
                inst_times_su.extend(times[inst]["M"])
                inst_times_su_sm.extend(times[inst]["M"])
            if "L" in times[inst]:
                inst_times_su.extend(times[inst]["L"])
    print "S,M,L #annotators md:", len(inst_times_md)
    print "S,M,L #annotators times su:", len(inst_times_su)
    print "S,M #annotators md:", len(inst_times_md_sm)
    print "S,M #annotators times su:", len(inst_times_su_sm)
    print inst_times_md
    times_md = np.array(inst_times_md, dtype=float)
    times_su = np.array(inst_times_su, dtype=float)
    times_md_sm = np.array(inst_times_md_sm, dtype=float)
    times_su_sm = np.array(inst_times_su_sm, dtype=float)
    # print "S,M,L shape MD", times_md.shape

    # Store n medians; i-th entry corresponds to the annotation time
    # of the i-th tweet averaged over all annotators.
    # Compute median over columns ignoring entries with nan
    # avg_times = np.nanmean(times, axis=0).T
    median_times_md = np.nanmedian(times_md, axis=0).T
    median_times_su = np.nanmedian(times_su, axis=0).T
    median_times_md_sm = np.nanmedian(times_md_sm, axis=0).T
    median_times_su_sm = np.nanmedian(times_su_sm, axis=0).T

    #############################
    # Compute medians per group #
    #############################
    median_group_times_md = {}
    median_group_times_su = {}
    for inst in times:
        # Compute medians per group
        for group in times[inst]:
            print "group: {} annos: {}".format(group, len(times[inst][group]))
            print times[inst][group]
            group_times = np.array(times[inst][group], dtype=float)
            print group_times
            medians = np.nanmedian(group_times, axis=0).T
            if inst == "md":
                print "md"
                median_group_times_md[group] = medians
            else:
                print "su"
                median_group_times_su[group] = medians

    ############################################
    # Fit a polynomial to the annotation times #
    ############################################
    # Degree of the polynomial
    k = 3
    # Try out different combinations of where the learning effect ends
    for i in xrange(5, n, 1):
        # We use 2 intervals for fitting and 2 entries per interval describe
        # lower and upper bound
        intervals = [[0, i], [i, to_keep-1]]
        # 1. Plot institutions
        #######################
        print "fit till n=", i
        # Important: don't change the first 2 parameters of the output file name
        # as plot_acceleration() needs to extract these!!!

        # Plot the data for institutions with all annotator groups
        pure_fname = "md_{}_{}_acceleration_till_{}_{}.pdf"\
            .format(to_keep, "_all_", i, dataset_type)
        acc_fname = "md_{}_{}_acceleration_fit_polynomial_degree_{}_till_{}_" \
                    "{}.pdf".format(to_keep, "_all_", k, i, dataset_type)
        annos = times_md.shape[0]
        print "annos", annos
        plot_acceleration(median_times_md, annos, os.path.join(dst, pure_fname),
                          os.path.join(dst, acc_fname), k, intervals)

        annos = times_su.shape[0]
        pure_fname = "su_{}_{}_acceleration_till_{}_{}.pdf"\
            .format(to_keep, "_all_", i, dataset_type)
        acc_fname = "su_{}_{}_acceleration_fit_polynomial_degree_{}_till_{}_" \
                    "{}.pdf".format(to_keep, "_all_", k, i, dataset_type)
        plot_acceleration(median_times_su, annos, os.path.join(dst, pure_fname),
                          os.path.join(dst, acc_fname), k, intervals)

        # Plot the data for institutions with S and M groups
        annos = times_md_sm.shape[0]
        pure_fname = "md_{}_{}_acceleration_till_{}_{}.pdf"\
            .format(to_keep, "_sm_", i, dataset_type)
        acc_fname = "md_{}_{}_acceleration_fit_polynomial_degree_{}_till_{}_" \
                    "{}.pdf".format(to_keep, "_sm_", k, i, dataset_type)
        plot_acceleration(median_times_md_sm, annos,
                          os.path.join(dst, pure_fname),
                          os.path.join(dst, acc_fname), k, intervals)

        annos = times_su_sm.shape[0]
        pure_fname = "su_{}_{}_acceleration_till_{}_{}.pdf"\
            .format(to_keep, "_sm_", i, dataset_type)
        acc_fname = "su_{}_{}_acceleration_fit_polynomial_degree_{}_till_{}_" \
                    "{}.pdf".format(to_keep, "_sm_", k, i, dataset_type)
        plot_acceleration(median_times_su_sm, annos,
                          os.path.join(dst, pure_fname),
                          os.path.join(dst, acc_fname), k, intervals)
        # if i == 5:
        #     break
        # 2. Plot groups
        # ################
        # S
        annos = len(times["md"]["S"])
        pure_fname = "md_{}_{}_acceleration_till_{}_{}.pdf"\
            .format(to_keep, "_s_", i, dataset_type)
        acc_fname = "md_{}_{}_acceleration_fit_polynomial_degree_{}_till_{}_" \
                    "{}.pdf".format(to_keep, "_s_", k, i, dataset_type)
        plot_acceleration(median_group_times_md["S"], annos,
                          os.path.join(dst, pure_fname),
                          os.path.join(dst, acc_fname), k, intervals)

        annos = len(times["su"]["S"])
        pure_fname = "su_{}_{}_acceleration_till_{}_{}.pdf"\
            .format(to_keep, "_s_", i, dataset_type)
        acc_fname = "su_{}_{}_acceleration_fit_polynomial_degree_{}_till_{}_" \
                    "{}.pdf".format(to_keep, "_s_", k, i, dataset_type)
        plot_acceleration(median_group_times_su["S"], annos,
                          os.path.join(dst, pure_fname),
                          os.path.join(dst, acc_fname), k, intervals)

        # M
        pure_fname = "md_{}_{}_acceleration_till_{}_{}.pdf"\
            .format(to_keep, "_m_", i, dataset_type)
        acc_fname = "md_{}_{}_acceleration_fit_polynomial_degree_{}_till_{}_" \
                    "{}.pdf".format(to_keep, "_m_", k, i, dataset_type)
        plot_acceleration(median_group_times_md["M"], annos,
                          os.path.join(dst, pure_fname),
                          os.path.join(dst, acc_fname), k, intervals)

        annos = len(times["su"]["S"])
        pure_fname = "su_{}_{}_acceleration_till_{}_{}.pdf"\
            .format(to_keep, "_m_", i, dataset_type)
        acc_fname = "su_{}_{}_acceleration_fit_polynomial_degree_{}_till_{}_" \
                    "{}.pdf".format(to_keep, "_m_", k, i, dataset_type)
        plot_acceleration(median_group_times_su["M"], annos,
                          os.path.join(dst, pure_fname),
                          os.path.join(dst, acc_fname), k, intervals)
        # L
        pure_fname = "md_{}_{}_acceleration_till_{}_{}.pdf"\
            .format(to_keep, "_l_", i, dataset_type)
        acc_fname = "md_{}_{}_acceleration_fit_polynomial_degree_{}_till_{}_" \
                    "{}.pdf".format(to_keep, "_l_", k, i, dataset_type)
        plot_acceleration(median_group_times_md["L"], annos,
                          os.path.join(dst, pure_fname),
                          os.path.join(dst, acc_fname), k, intervals)

        annos = len(times["su"]["S"])
        pure_fname = "su_{}_{}_acceleration_till_{}_{}.pdf"\
            .format(to_keep, "_l_", i, dataset_type)
        acc_fname = "su_{}_{}_acceleration_fit_polynomial_degree_{}_till_{}_" \
                    "{}.pdf".format(to_keep, "_l_", k, i, dataset_type)
        plot_acceleration(median_group_times_su["L"], annos,
                          os.path.join(dst, pure_fname),
                          os.path.join(dst, acc_fname), k, intervals)


def plot_acceleration(y, annos, pure_fpath, acc_fpath, k, intervals):
    """
    Plots acceleration graph.

    Parameters
    ----------
    y: list of float - each float represents a median annotation time in the
    dataset.
    annos: int - number of annotators in the dataset.
    pure_fpath: str - path where the scatter plot without acceleration should
    be stored.
    acc_fpath: str - path where the scatter plot with fitted acceleration should
    be stored.
    k: int - degree of the polynomial to be fit to the data in order to
    approximate the acceleration.
    intervals: list of intervals - [[l,u], [l,u]], each entry represents the
    (l)ower and (u)pper bounds of an interval.

    """
    # MD, SU, or something else
    institution = pure_fpath.split("_")[4]
    # Number of tweets to be kept
    try:
        TO_KEEP = int(pure_fpath.split("_")[-7])
    except ValueError:
        # Annotator should be plotted, so his/her name is also part of the file
        # name
        TO_KEEP = int(pure_fpath.split("_")[-8])
    # Plotting
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    s = [10 for n in range(annos)]
    # Use different colors to encode the annotator group of each participant
    x = range(len(y))
    # ax.scatter(x, y, color="silver", s=s, label="Median annotation time")
    ax.scatter(x, y, color="darkgray", s=s)
    # Hide the right and top spines (lines)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Title
    # if len(group) > 0:
    #     title = "Median annotation time for {} in {}"\
    #         .format(institution.upper(), group.title())
    # elif len(name) > 0:
    #     title = "Annotation times for {}".format(name.title())
    # else:
    #     title = "Median annotation time for {}".format(institution.upper())
    # plt.title(title)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # Set labels of axes
    ax.set_xlabel("i-th annotated tweet")
    ax.set_ylabel("Median annotation time in s")
    # Add legend outside of plot
    legend = ax.legend(loc="best", shadow=True)
    # Limits of axes
    plt.xlim(-1, TO_KEEP + 3)
    plt.ylim(-5, max(y) + 3)
    plt.savefig(pure_fpath, bbox_inches='tight', dpi=600)
    # Intervals for which separate derivatives should be calculated
    # intervals = [[0, 20], [20, 40], [40, 60]]
    # intervals = [[0, 30]]
    # intervals = [[0, 25], [25, 59]]
    # intervals = [[0, 19], [19, 59]]
    COLORS_P = ["black", "black", "black"]
    COLORS_D = ["darkorange", "darkorange", "darkorange"]

    # Use numpy's new polynomial module to fit the line with a polynomial
    # of degree k. Fit it only to the current interval
    # http://stackoverflow.com/questions/18767523/fitting-data-with-numpy

    # Fit polynomial of 3rd degree to the data
    for idx, (l, o) in enumerate(intervals):
        # print "Fit interval {}-{}".format(l, o)
        xp = np.linspace(l, o, 1000)
        # x/y have either 50 values, then 100 NANs or 150 values
        # For polyfit to work, we must simply remove the NANs
        x_cur = np.array(x[l:o])
        y_cur = np.array(y[l:o])
        # Indices where x and y are not NAN
        nan_idx = np.isfinite(x_cur) & np.isfinite(y_cur)
        f = poly.polyfit(x_cur[nan_idx], y_cur[nan_idx], k)
        # print "f=", f
        yp = poly.polyval(xp, f)
        # Plot polynomial
        # Uncomment for k=3
        # ax.plot(xp, yp, "-", color=COLORS_P[idx], linewidth=2,
        #         label="f={:<4.1f}x^3+{:<4.1f}x^2+{:<4.1f}x+{:<4.1f}"
        #         .format(f[3], f[2], f[1], f[0]))
        if idx == 0:
            ax.plot(xp, yp, "-", color=COLORS_P[idx], linewidth=2,
                    label="Fitted polynomial")
        else:
            ax.plot(xp, yp, "-", color=COLORS_P[idx], linewidth=2)
        # Uncomment for k=4
        # ax.plot(xp, yp, "-", color=COLORS_P[idx], linewidth=2,
        #         label="{:<4.1f}x^4+{:<4.1f}x^3+{:<4.1f}x^2+{:<4.1f}x+{:<4.1f}"
        #         .format(f[4], f[3], f[2], f[1], f[0]))

        # http://stackoverflow.com/questions/29634217/get-minimum-points-of-numpy-poly1d-curve
        # Coefficients must be past from highest order to lowest order, the
        # opposite of what polyfit() returns, so reverse it
        p = np.poly1d(f[::-1])
        # print "p", p
        pi = p.deriv()
        # print p
        # print "p'", pi
        pii = p.deriv(2)
        # piii = p.deriv(3)
        # print "p''", pii
        roots = pi.r
        # print "roots", roots
        # f' = 0 yields critical points where slope is 0
        # r_crit = roots[roots.imag == 0].real
        r_crit = roots.real
        # print "crit", r_crit
        # Select from critical points the minima, where f'' > 0
        # compute local minima excluding range boundaries
        # x_min = r_crit[pii(r_crit) > 0]
        x_min = r_crit
        y_min = p(x_min)
        # print "x_min", x_min
        # print "y_min", y_min
        # ax.plot(x_min, y_min, "o", color="red")

        # Add y-offset to first derivative, namely the difference of the
        # computed value and y_min
        # If there are multiple points, choose the one with minimum y-value
        min_idx = np.argmin(y_min)
        if not isinstance(y_min, np.float64):
            y_min = y_min[min_idx]
        # print "min y at pos", min_idx
        # print "corresponding x:", x_min[min_idx]
        if not isinstance(x_min, np.float64):
            x_min = x_min[min_idx]
        # print "difference", y_min - pi(x_min)
        # Add offset to the constant of the equation
        pi.c[2] += y_min - pi(x_min)
        ypi = pi(xp)

        # Don't display first derivative
        # Uncomment for k=3
        # ax.plot(xp, ypi, "--", color="black", linewidth=2,
        #         label="f'={:<4.1f}x^2+{:<4.1f}x+{:<4.1f}"
        #         .format(pi[0], pi[1], pi[2]))
        # Uncomment for k=4
        # ax.plot(xp, ypi, "--", color="black", linewidth=2,
        #         label="{:<4.1f}x^3+{:<4.1f}x^2+{:<4.1f}x+{:<4.1f}"
        #         .format(pi[3], pi[2], pi[1], pi[0]))

        # Plot 2nd derivative
        # x_min will be the same as in pi
        roots = pii.r
        # print "roots", roots
        # f' = 0 yields critical points where slope is 0
        # r_crit = roots[roots.imag == 0].real
        r_crit = roots.real
        # print "crit", r_crit
        # Select from critical points the minima, where f''' > 0
        # compute local minima excluding range boundaries
        # x_min = r_crit[piii(r_crit) > 0]
        y_min = p(x_min)
        # print "x_min", x_min
        # print "y_min", y_min
        # ax.plot(x_min, y_min, "o", color="green")

        # Add y-offset to second derivative, namely the difference of the
        # computed value and y_min
        # If there are multiple points, choose the one with minimum y-value
        min_idx = np.argmin(y_min)
        # print "min y at pos", min_idx
        if not isinstance(y_min, np.float64):
            y_min = y_min[min_idx]
            # print "corresponding x:", x_min[min_idx]
        if not isinstance(x_min, np.float64):
            x_min = x_min[min_idx]
        # print "difference", y_min - pi(x_min)
        # Add offset to the constant of the equation
        pii.c[1] +=  pi(x_min)
        # print "p'(y_min)", y_min
        ypii = pii(xp)
        # Uncomment for k=3
        # ax.plot(xp, ypii, "-", color=COLORS_D[idx], linewidth=3,
        #         label="f''={:<4.1f}x+{:<4.1f}".format(pii[1], pii[0]))
        if idx == 0:
            ax.plot(xp, ypii, "-", color=COLORS_D[idx], linewidth=2,
                    label="Acceleration")
        else:
            ax.plot(xp, ypii, "-", color=COLORS_D[idx], linewidth=2)
        # Uncomment for k=4
        # ax.plot(xp, ypii, "-", color=COLORS_D[idx], linewidth=2,
        #         label="{:<4.1f}x^2+{:<4.1f}x+{:<4.1f}".format(pii[2], pii[1],
            # pii[0]))
    # Split point
    s = intervals[0][1]
    # Add horizontal line to indicate where we split the intervals
    ax.plot((s, s), (0, ax.get_ylim()[1]), "--", color="red", linewidth=2)
    plt.ylim(0, ax.get_ylim()[1])
    # ax.legend(loc="upper right", shadow=True, bbox_to_anchor=(1, 1.4))
    ax.legend(loc="best", shadow=True, fontsize=FONTSIZE)
    plt.savefig(acc_fpath, bbox_inches='tight', dpi=600)
    plt.close()


def get_total_anno_times(tweet_path, anno_path, n, cleaned=False):
    """
    Computes total annotation time per annotator for all her tweets and
    store those times per annotator group. Keeps n tweets per annotator.

    Parameters
    ----------
    tweet_path: str - path to tweet csv dataset.
    anno_path: str - path to annotator csv dataset.
    n: int - number of tweets to keep per annotator.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing median
    annotation times.

    Returns
    -------
    dict.
    {
        "S": [[total times by anno1], [total times by anno2], ...],
        "M": [[...], [...], ...],
        "L": [[...], [...], ...]
    }

    """
    # Group is key, list are annotation times of all annotators of that group
    times = {"S": [],
             "M": [],
             "L": []
             }
    data = pseudo_db.Data(tweet_path, anno_path)
    # For each annotator
    for anno in data.annotators.all_annos():
        anno_times = []
        group = anno.get_group()
        # For each tweet
        for t in anno.get_labeled_tweets():
            labels = t.get_anno_labels()
            a_times = t.get_anno_times()
            # Annotation times for hierarchy levels 2 and 3 might be ignored
            t2 = pseudo_db.ZERO
            t3 = pseudo_db.ZERO
            # First level
            rel_label = labels[0]
            t1 = a_times[0]
            # Discard remaining labels if annotator chose "Irrelevant"
            # Consider other sets of labels iff either the cleaned
            # dataset should be created and the label is "relevant" OR
            # the raw dataset should be used.
            if (cleaned and rel_label != "Irrelevant") or not cleaned:
                l2 = labels[1]
                t2 = a_times[1]
                # Annotator labeled the 3rd set of labels as well
                if l2 == pseudo_db.GROUND_TRUTH["Non-factual"]:
                    t3 = a_times[2]
            total = sum([t1, t2, t3])
            anno_times.append(total)
        # Fill up the list with "None" values if less than <n> values are
        # available for an annotator
        anno_times.extend([None] * (n - len(anno_times)))
        assert (len(anno_times[:n]) == n)
        times[group].append(anno_times[:n])
    return times


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))

    # Path to the tweets of MD
    md_tweets = os.path.join(base_dir, "anonymous",
                             "tweets_hierarchical_md.csv")
    # Path to the annotators of MD
    md_annos = os.path.join(base_dir, "anonymous",
                            "annotators_hierarchical_md.csv")
    # Path to the tweets of SU
    su_tweets = os.path.join(base_dir, "anonymous",
                             "tweets_hierarchical_su.csv")
    # Path to the annotators of SU
    su_annos = os.path.join(base_dir, "anonymous",
                            "annotators_hierarchical_su.csv")

    FIG_DIR = os.path.join(base_dir, "results", "figures",
                           "acceleration_median_annotation_time")
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    n = 47
    # Keep the first <to_keep> tweets per annotator
    to_keep = 50
    # Raw
    plot_learning_acceleration(FIG_DIR, n, to_keep, md_tweets, md_annos,
                               su_tweets, su_annos, cleaned=False)
    # Cleaned
    plot_learning_acceleration(FIG_DIR, n, to_keep, md_tweets, md_annos,
                               su_tweets, su_annos, cleaned=True)
    to_keep = 150
    # Raw
    plot_learning_acceleration(FIG_DIR, n, to_keep, md_tweets, md_annos,
                               su_tweets, su_annos, cleaned=False)
    # Cleaned
    plot_learning_acceleration(FIG_DIR, n, to_keep, md_tweets, md_annos,
                               su_tweets, su_annos, cleaned=True)
