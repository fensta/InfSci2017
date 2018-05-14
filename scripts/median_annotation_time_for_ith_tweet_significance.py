"""
Plots the median annotation times in an institution for the i-th tweet, i.e.
median annotation times it took annotators to assign a label to their 1st, 2nd,
3rd... tweet. Done so including annotators from L as well as excluding them.
When fitting a line, the statistics are also stored in a txt file.
"""
import os
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from numpy import array

import pseudo_db


def plot_median_anno_time_for_ith_tweet(
        tweet_path, anno_path, fig_dir, stat_dir, cleaned=False, n=50,
        with_l=True):
    """
    Plots how the median annotation times for the i-th tweet in an institution.
    Also fits a line to these times indicating the trend of how the times
    develop if more tweets are labeled by annotators.

    Parameters
    ----------
    tweet_path: str - path to tweet csv dataset.
    anno_path: str - path to annotator csv dataset.
    fig_dir: str - directory in which the plots will be stored.
    stat_dir: str - directory in which the stats will be stored.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing average
    annotation times.
    n: int - maximum number of tweets that should be discarded from the
    beginning of an annotators median annotation times due
    to the learning effect.
    with_l: True if annotators of L should also be used.

    """
    # n will be our index; since it's 0-based, add 1
    n += 1
    # Find out for which institution this plot is - su or md
    data = pseudo_db.Data(tweet_path, anno_path)
    institution = data.annotators.get_ith_annotator(0).get_dataset_name()
    # institution = "md"
    # for db_idx in db_idxs:
    #     db = dbs[db_idx]
    #     if db == "turannotationtool":
    #         institution = "su"
    #         break
    dataset_type = "raw"
    if cleaned:
        dataset_type = "cleaned"
    size = "without_l"
    if with_l:
        size = "with_l"
    times = get_total_anno_times(tweet_path, anno_path, cleaned=cleaned,
                                 with_l=with_l)
    # m x n matrix that stores n tweets for each of the m annotators. Thus,
    # each row represents the annotation times of a single annotator.
    # If values are missing, np.nan is used. Since the maximum number of
    # annotated tweets is 500, each row contains 500 columns
    times = np.array(times, dtype=float)

    # Test with different k, i.e. leave out the first k tweets before computing
    # median annotation times per annotator
    for k in xrange(n):
        # print "k = {}".format(k)
        # Remove first k tweets (= first k columns) from each annotator (= row)
        removed = times[:, k:]
        # Store m medians; i-th column corresponds to the median annotation time
        # of the i-th tweet in that institution after removing the first k
        # tweets
        # Compute medians over columns ignoring entries with nan
        y = np.nanmedian(removed, axis=0).T
        annos = times.shape[0]
        s = [10 for b in range(annos)]
        x = range(len(y))

        name = "{}_ith_tweet_median_annotation_time_ignore_first_{}_tweets_" \
               "{}_{}".format(institution, k, size, dataset_type)
        tab_dst = os.path.join(stat_dir, name + ".txt")
        fig_dst = os.path.join(fig_dir, name + ".png")

        # Fit a line to the times, f(x) = mx + n
        xf, yf, m = _lin_fit(x, y, tab_dst)

        # Create plot
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(111)
        ax.scatter(x, y, color="red", s=s, label="Median annotation time")
        # Plot the fit
        ax.plot(xf, yf, color="black", linewidth=2, label="slope={:4.2f}"
                .format(m))
        # Hide the right and top spines (lines)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Title - k+1 because we start with k=0
        title = "Median annotation times for {} ignoring the first {} tweets"\
            .format(institution.upper(), k)
        # Set title and increase space to plot
        plt.title(title)
        ttl = ax.title
        ttl.set_position([.5, 1.03])
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # Set labels of axes
        ax.set_xlabel("i-th annotated tweet")
        ax.set_ylabel("Median annotation time in s")
        # Add legend outside of plot
        legend = ax.legend(shadow=True, bbox_to_anchor=(1, 1.5))
        # Limits of axes
        plt.xlim(-1, ax.get_xlim()[1])
        plt.ylim(0, ax.get_ylim()[1])
        plt.savefig(fig_dst, bbox_inches='tight', dpi=300)
        plt.close()


def get_total_anno_times(tweet_path, anno_path, cleaned=False, with_l=False):
    """
    Computes total annotation time per annotator for all her tweets and
    store those times. Keeps n tweets per annotator.

    Parameters
    ----------
    tweet_path: str - path to tweet csv dataset.
    anno_path: str - path to annotator csv dataset.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing median
    annotation times.
    with_l: True if annotators of L should also be used.

    Returns
    -------
    List of lists of floats.
    Each inner list represents all annotation times of a single annotator.
    Hence, the len(outer list) = number of annotators in institution. Note that
    each inner list contains 500 times, if an annotator labeled less, the
    missing values are filled with np.nan.

    """
    # Maximum number of tweets annotated by a single person -> add dummy values
    # (np.nan) if an annotator labeled less tweets
    n = 500
    # Without L there are only 150 tweets labeled at most
    if not with_l:
        n = 150
    # Group is key, list are annotation times of all annotators of that group
    times = []
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
        if with_l or group != "L":
            # Make sure that each list contains exactly <n> entries
            assert (len(anno_times) == n)
            times.append(anno_times)
    return times


# http://glowingpython.blogspot.com.tr/2012/03/linear-regression-with-numpy.html
def _lin_fit(x, y, dst):
    """
    Performs linear regression.

    Parameters
    ----------
    x: list - x-values.
    y: list - y-values.
    dst: str - path where output will be stored.

    Returns
    -------
    list, list, double.
    List of x-values, list of y-values representing the fitted line; slope of
    the fitted line.

    """
    # Explanation what the null hypothesis is (slope = 0):
    # http://stattrek.com/regression/slope-test.aspx?Tutorial=AP
    slope, intercept, corr_coeff, p_value, std_err = stats.linregress(array(x),
                                                                      array(y))
    # print "r^2: {}, i.e. {}% of the randomness in the data is explained by " \
    #       "the fitted line.".format(corr_coeff**2, corr_coeff**2 * 100)
    # print "p-value:", p_value
    # print "Standard error of the estimated gradient:", std_err
    # print "Slope:", slope
    line = slope*array(x) + intercept
    with open(dst, "wb") as f:
        write_significance(f, p_value)
        f.write("r: {:.8f}\n".format(corr_coeff))
        r = corr_coeff**2
        f.write("explained randomness: {:.2f} {:.2f}\n".format(r, r * 100))
        f.write("standard_error: {:.8f}\n".format(std_err))
        f.write("slope: {:.3f}\n".format(slope))
    return x, line.tolist(), slope


def write_significance(f, p):
    """
    Write formatted significance infsci2017_results (statistic and p-value) given
    an open file handle. Also adds ** (or *) to the p-value if the p-value is
    significant on the 0.01 (or 0.05) significance level.

    Parameters
    ----------
    f: file handle - file in which the line should be written.
    computed.
    p: float - p-value.

    """
    if p < 0.01:
        f.write("p: {:.8f}**\n".format(p))
    elif 0.01 <= p < 0.05:
        f.write("p: {:.8f}*\n".format(p))
    else:
        f.write("p: {:.8f}\n".format(p))


def plot_p_values(root_dir, dst, cleaned=False, with_l=True):
    """
    Plots the computed p-values of the fitted lines.

    Parameters
    ----------
    root_dir: str - directory containing txt files with p-values.
    dst: str - path where plot will be stored.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing average
    annotation times.
    with_l: True if annotators of L should also be used.

    """
    # Tuples of (k, p-value) - this is necessary because we need to plot it
    # with ascending k's (= number of tweets that were ignored at the beginning
    # for each user to account for the learning effect)
    ps = []
    # Extract p-value from each file
    for fi in os.listdir(root_dir):
        to_open = False
        # print fi
        # Only consider files that actually contain p-values
        if "time_ignore_first" in fi:
            # Skip files that shouldn't be considered
            if cleaned and with_l:
                if "with_l" in fi and "_cleaned" in fi:
                    to_open = True
            if not cleaned and with_l:
                if "with_l" in fi and "_raw" in fi:
                    to_open = True
            if cleaned and not with_l:
                if "without_l" in fi and "_cleaned" in fi:
                    to_open = True
            if not cleaned and not with_l:
                if "without_l" in fi and "_raw" in fi:
                    to_open = True
            if to_open:
                k = int(fi.split("ignore_first_")[1].split("_")[0])
                f_path = os.path.join(root_dir, fi)
                with open(f_path, "rb") as f:
                    lines = f.readlines()
                for line in lines:
                    if line.startswith("p:"):
                        line = line.rstrip()
                        p = line.split(" ")[1]
                        if p.endswith("**"):
                            p = p[:-2]
                        elif p.endswith("*"):
                            p = p[:-1]
                        p = float(p)
                        ps.append((k, p))

    # Sort (k, p) tuples according to k
    ps = sorted(ps, key=itemgetter(0))
    # Find out when the significance level is exceeded
    for k, p in ps:
        if p >= 0.05:
            print "significance level of 0.05 exceeded when removing the " \
                  "first {} tweets".format(k)
            break
    y = [tup[1] for tup in ps]
    # Plot p-values
    x = range(len(ps))
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    s = [5 for i in y]
    ax.scatter(x, y, color="red", s=s, label="p-value")
    # Hide the right and top spines (lines)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Set title and increase space to plot
    title = "p-values of fitted line"
    plt.title(title)
    ttl = ax.title
    ttl.set_position([.5, 1.03])
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # Set labels of axes
    ax.set_xlabel("#ignored tweets at beginning")
    ax.set_ylabel("p-value for slopes")
    # Add legend outside of plot
    # legend = ax.legend(shadow=True, bbox_to_anchor=(1, 1.5))
    legend = ax.legend(shadow=True, loc="best")
    # Limits of axes
    plt.xlim(-1, ax.get_xlim()[1])
    plt.ylim(0, ax.get_ylim()[1])
    plt.savefig(dst, bbox_inches='tight', dpi=300)
    plt.close()


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

    # Maximum number of tweets to skip in the beginning to account for
    # the learning effect
    # 148 because we have 150 tweets if we ignore L and we need 2 points to
    # compute a slope
    n = 148
    # Directories in which figure/statistics will be stored
    FIG_DIR = os.path.join(base_dir, "results", "figures",
                           "ith_tweet_median_annotation_time_su_all")
    STAT_DIR = os.path.join(base_dir, "results", "stats",
                            "ith_tweet_median_annotation_time_su_all")
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    if not os.path.exists(STAT_DIR):
        os.makedirs(STAT_DIR)

    ########################################################################
    # 1.PLOT median annotation times for i-th tweet and compute significance
    ########################################################################
    # a) with L
    ###########
    # Raw - SU
    plot_median_anno_time_for_ith_tweet(
        su_tweets, su_annos, FIG_DIR, STAT_DIR, n=n, cleaned=False, with_l=True)
    # Cleaned - SU
    plot_median_anno_time_for_ith_tweet(
        su_tweets, su_annos, FIG_DIR, STAT_DIR, n=n, cleaned=True, with_l=True)

    FIG_DIR = os.path.join(base_dir, "results", "figures",
                           "ith_tweet_median_annotation_time_md_all")
    STAT_DIR = os.path.join(base_dir, "results", "stats",
                            "ith_tweet_median_annotation_time_md_all")
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    if not os.path.exists(STAT_DIR):
        os.makedirs(STAT_DIR)
    # Raw - MD
    plot_median_anno_time_for_ith_tweet(
        md_tweets, md_annos, FIG_DIR, STAT_DIR, n=n, cleaned=False, with_l=True)
    # Cleaned - MD
    plot_median_anno_time_for_ith_tweet(
        md_tweets, md_annos, FIG_DIR, STAT_DIR, n=n, cleaned=True, with_l=True)

    # b) without L
    ##############
    # Raw - SU
    FIG_DIR = os.path.join(base_dir, "results", "figures",
                           "ith_tweet_median_annotation_time_su_all")
    STAT_DIR = os.path.join(base_dir, "results", "stats",
                            "ith_tweet_median_annotation_time_su_all")
    plot_median_anno_time_for_ith_tweet(
        su_tweets, su_annos, FIG_DIR, STAT_DIR, n=n, cleaned=False,
        with_l=False)
    # Cleaned - SU
    plot_median_anno_time_for_ith_tweet(
        su_tweets, su_annos, FIG_DIR, STAT_DIR, n=n, cleaned=True, with_l=False)

    FIG_DIR = os.path.join(base_dir, "results", "figures",
                           "ith_tweet_median_annotation_time_md_all")
    STAT_DIR = os.path.join(base_dir, "results", "stats",
                            "ith_tweet_median_annotation_time_md_all")
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    if not os.path.exists(STAT_DIR):
        os.makedirs(STAT_DIR)
    # Raw - MD
    plot_median_anno_time_for_ith_tweet(
        md_tweets, md_annos, FIG_DIR, STAT_DIR, n=n, cleaned=False,
        with_l=False)
    # Cleaned - MD
    plot_median_anno_time_for_ith_tweet(
        md_tweets, md_annos, FIG_DIR, STAT_DIR, n=n, cleaned=True, with_l=False)

    ######################################
    # 2. Plot computed p-values of slopes
    ######################################
    # a) SU
    ########
    FIG_DIR = os.path.join(base_dir, "results", "figures",
                           "ith_tweet_median_annotation_time_su_all")
    STAT_DIR = os.path.join(base_dir, "results", "stats",
                            "ith_tweet_median_annotation_time_su_all")
    print "SU, raw, L:"
    dst = os.path.join(FIG_DIR, "su_slope_p_values_raw_with_l.png")
    plot_p_values(STAT_DIR, dst, cleaned=False, with_l=True)
    print "SU, cleaned, L:"
    dst = os.path.join(FIG_DIR, "su_slope_p_values_cleaned_with_l.png")
    plot_p_values(STAT_DIR, dst, cleaned=True, with_l=True)
    print "SU, raw, no L:"
    dst = os.path.join(FIG_DIR, "su_slope_p_values_raw_without_l.png")
    plot_p_values(STAT_DIR, dst, cleaned=False, with_l=False)
    print "SU, cleaned, no L:"
    dst = os.path.join(FIG_DIR, "su_slope_p_values_cleaned_without_l.png")
    plot_p_values(STAT_DIR, dst, cleaned=True, with_l=False)

    # # b) MD
    # ########
    FIG_DIR = os.path.join(base_dir, "results", "figures",
                           "ith_tweet_median_annotation_time_md_all")
    STAT_DIR = os.path.join(base_dir, "results", "stats",
                            "ith_tweet_median_annotation_time_md_all")
    dst = os.path.join(FIG_DIR, "md_slope_p_values_raw_with_l.png")
    print "MD, raw, L:"
    plot_p_values(STAT_DIR, dst, cleaned=False, with_l=True)
    print "MD, cleaned, L:"
    dst = os.path.join(FIG_DIR, "md_slope_p_values_cleaned_with_l.png")
    plot_p_values(STAT_DIR, dst, cleaned=True, with_l=True)
    print "MD, raw, no L:"
    dst = os.path.join(FIG_DIR, "md_slope_p_values_raw_without_l.png")
    plot_p_values(STAT_DIR, dst, cleaned=False, with_l=False)
    print "MD, cleaned, no L:"
    dst = os.path.join(FIG_DIR, "md_slope_p_values_cleaned_without_l.png")
    plot_p_values(STAT_DIR, dst, cleaned=True, with_l=False)
