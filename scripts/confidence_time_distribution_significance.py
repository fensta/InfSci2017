"""
Plots the median confidence times per
- group in an institution
- institution
Likewise, computes the significance level for these 2 possibilities.
"""

import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import pseudo_db


# For Information sciences
FONTSIZE = 12
plt.rcParams.update({'font.size': FONTSIZE})


def analyze_distribution(md_tw_p, md_a_p, su_tw_p, su_a_p, fig_dst, stat_dst,
                         cleaned=False):
    """
    Plots median confidence times by
    a) institution
    b) group per institution.
    Also computes the significance of a) and b) with and without annotators from
    group L (because only a few annotators were part of that
    # group, meaning the infsci2017_results could get biased as 350 tweets were labeled
    # only by 1 (for MD) or 3 (for SU) annotators).

    Parameters
    ----------
    md_tw_p: str - path to MD tweets dataset in csv format.
    md_a_p: str - path to MD annotators dataset in csv format.
    su_tw_p: str - path to SU tweets dataset in csv format.
    su_a_p: str - path to SU annotators dataset in csv format.
    fig_dst: str - directory in which the plot will be stored.
    stat_dst: str - directory in which the statistics will be stored.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing median
    confidence times.

    """
    agg_type = "raw"
    if cleaned:
        agg_type = "cleaned"

    # ##############################
    # Computations including L and M
    # ##############################
    # Get data grouped by group per institution
    times = get_times_from_dbs(md_tw_p, md_a_p, su_tw_p, su_a_p,
                               cleaned=cleaned)
    fname = "all_median_conf_times_with_l_with_m"
    fpath_plot = os.path.join(fig_dst, fname + "_{}.pdf"
                              .format(agg_type))
    title = "{} median confidence times per institution".format(
        agg_type.title())

    # Plot all institutions in one plot
    _plot_distribution(times, title, fpath_plot)
    # Test statistical significance of all institutions and groups
    stat_path = os.path.join(stat_dst, fname + "_{}.txt".format(agg_type))
    compute_significance(times, stat_path)

    # Plot MD only
    fname = "md_median_conf_times_with_l_with_m"
    fpath_plot = os.path.join(fig_dst, fname + "_{}.pdf".format(agg_type))
    times_cpy = deepcopy(times)
    del times_cpy["su"]
    _plot_distribution(times_cpy, title, fpath_plot)

    # Plot SU only
    fname = "su_median_conf_times_with_l_with_m"
    fpath_plot = os.path.join(fig_dst, fname + "_{}.pdf".format(agg_type))
    times_cpy = deepcopy(times)
    del times_cpy["md"]
    _plot_distribution(times_cpy, title, fpath_plot)

    # #########################################################################
    # Computations excluding L (only affects label distribution of institution)
    # #########################################################################
    fname = "all_median_conf_times_without_l_with_m"
    fpath_plot = os.path.join(fig_dst, fname + "_{}.pdf".format(agg_type))
    title = "{} median confidence times per institution (without L)".format(
        agg_type.title())

    # Remove "L" from all institutions
    for inst, groups in times.iteritems():
        if "L" in groups:
            del times[inst]["L"]

    # Plot all institutions in one plot
    _plot_distribution(times, title, fpath_plot)
    # Test statistical significance of all institutions and groups
    stat_path = os.path.join(stat_dst, fname + "_{}.txt".format(agg_type))
    compute_significance(times, stat_path)

    # Plot MD only
    fname = "md_median_conf_times_without_l_with_m"
    fpath_plot = os.path.join(fig_dst, fname + "_{}.pdf".format(agg_type))
    times_cpy = deepcopy(times)
    del times_cpy["su"]
    _plot_distribution(times_cpy, title, fpath_plot)

    # Plot SU only
    fname = "su_median_conf_times_without_l_with_m"
    fpath_plot = os.path.join(fig_dst, fname + "_{}.pdf".format(agg_type))
    times_cpy = deepcopy(times)
    del times_cpy["md"]
    _plot_distribution(times_cpy, title, fpath_plot)


def get_times_from_dbs(md_tw_p, md_a_p, su_tw_p, su_a_p, cleaned=False):
    """

        Gets the median annotation times from the DB per group of an instiution.

        Parameters
        ----------
        md_tw_p: str - path to MD tweets dataset in csv format.
        md_a_p: str - path to MD annotators dataset in csv format.
        su_tw_p: str - path to SU tweets dataset in csv format.
        su_a_p: str - path to SU annotators dataset in csv format.
        cleaned: bool - True if the data should be cleaned, i.e. if tweet is
        "irrelevant", its remaining labels are ignored for computing median
        annotation times.

        Returns
        -------
        dict.
        First dict represents median annotation times per group per institution
        ("md", "su","later" (if not None)) with the institutions as keys and dict as
        value which uses the groups ("S", "M", "L") as key and as value the list
        of all median annotation times for the given group in that institution. Note
        that groups without annotation times are deleted.

        """
    groups = {
        "md": get_median_conf_times(md_tw_p, md_a_p, cleaned),
        "su": get_median_conf_times(su_tw_p, su_a_p, cleaned),
    }
    print groups
    # Remove empty groups
    updated = deepcopy(groups)
    for inst, grps in groups.iteritems():
        for group in grps:
            times = len(groups[inst][group])
            # Keep only groups with more than 0 annotation times
            if times == 0:
                del updated[inst][group]

    return updated


def get_median_conf_times(tweet_path, anno_path, cleaned=False):
    """
    Computes median confidence time per annotator for all her tweets and
    store those times per annotator group.

    Parameters
    ----------
    tweet_path: str - path to tweet csv dataset.
    anno_path: str - path to annotator csv dataset.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing median
    annotation times.

    Returns
    -------
    dict.
    {
        "S": [median time by anno1, median time by anno2,...],
        "M": [median time by anno1, median time by anno2,...],
        "L": [median time by anno1, median time by anno2,...]
    }

    """
    res = {}
    data = pseudo_db.Data(tweet_path, anno_path)
    # For each annotator
    for anno in data.annotators.all_annos():
        anno_times = []
        group = anno.get_group()
        if group not in res:
            res[group] = []
        # For each tweet
        for t in anno.get_labeled_tweets():
            labels = t.get_anno_labels()
            times = t.get_conf_times()
            # Annotation times for hierarchy levels 2 and 3 might be ignored
            t2 = pseudo_db.ZERO
            t3 = pseudo_db.ZERO
            # First level
            rel_label = labels[0]
            t1 = times[0]
            # Discard remaining labels if annotator chose "Irrelevant"
            # Consider other sets of labels iff either the cleaned
            # dataset should be created and the label is "relevant" OR
            # the raw dataset should be used.
            if (cleaned and rel_label != "Irrelevant") or not cleaned:
                l2 = labels[1]
                t2 = times[1]
                # Annotator labeled the 3rd set of labels as well
                if l2 == pseudo_db.GROUND_TRUTH["Non-factual"]:
                    t3 = times[2]
            total = sum([t1, t2, t3])
            anno_times.append(total)
        times = np.array(anno_times, dtype=float)
        median = np.nanmedian(times, axis=0).T
        # Store median annotation time
        res[group].append(median)
    return res


def _plot_distribution(times, title, fpath):
    """
    Plots a scatter plot with the median confidence times. It plots the median
    times of all institutions present in the data.

    Parameters
    ----------
    times: dict of dict of list of float - key is institution ("md", "su",
    "later") and value is the a dict representing a group ("S", "M", "L") and
    its median confidence times in a list as value.
    title: str - title of the plot.
    fpath: str - path where the plot is stored.

    """
    # Find out which groups were used in the institutions
    groups = []
    # Name of institution with max keys
    for institution, groupss in times.iteritems():
        for group in groupss:
            if group not in groups:
                groups.append(group)
    # Sort entries deterministically
    legend_labels = sorted(groups, reverse=True)
    # Marker styles
    m = ["o", "x", "s"]
    # Color for each annotator group S, M, L
    colors = ["orangered", "deepskyblue", "yellowgreen"]
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    # Cache maximum y-value in graph
    max_y = 0
    for inst, (institution, groups) in enumerate(times.iteritems()):
        # Offset == number of median times that are already plotted
        offset = 0
        for col, (group, median_times) in enumerate(groups.iteritems()):
            # Plot mediantimes per annotator group (don't forget to add offset
            # for previously plotted median times
            x = [offset + i+1 for i in range(len(median_times))]
            ax.scatter(x, median_times, color=colors[col], marker=m[inst],
                       facecolors="none")
            offset += len(x)
            max_y = max(max_y, max(median_times))
    # plt.xlim(0, x[-1] + 1)
    # plt.ylim(9, max_y + 1)
    plt.xlim(0, ax.get_xlim()[1])
    # Create legend manually:
    # Get artists and labels for legend and chose which ones to display
    handles, labels = ax.get_legend_handles_labels()

    # Create custom artists
    custom_artists = []
    # Add groups
    for i, group in enumerate(legend_labels):
        artist = plt.Line2D((0, 1), (0, 0), color=colors[i])
        custom_artists.append(artist)
    # Add institution names
    for i, inst in enumerate(times):
        artist = plt.Line2D((0, 1), (0, 0), marker=m[i],
                            linestyle="", markerfacecolor="none",
                            markeredgecolor="black", markeredgewidth=1)
        custom_artists.append(artist)
        legend_labels.append(inst.upper())

    # Create legend from custom artist/label lists
    # ax.legend([handle for handle in handles] + custom_artists,
    #           [label for label in labels] + legend_labels, shadow=True,
    #           bbox_to_anchor=(1.4, 1))
    ax.legend([handle for handle in handles] + custom_artists,
              [label for label in labels] + legend_labels, shadow=True,
              bbox_to_anchor=(1, 1), fontsize=FONTSIZE)

    # Hide the right and top spines (lines)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # Set labels of axes
    ax.set_xlabel("Annotator ID")
    ax.set_ylabel("Median confidence time in s")
    # Set title and increase space to plot
    # plt.title(title)
    ttl = ax.title
    ttl.set_position([.5, 1.03])
    # Save plot
    plt.savefig(fpath, bbox_inches='tight', dpi=600)
    plt.close()


def compute_significance(labels, stat_path):
    """
    Computes if there is a statistically significant difference in the median
    confidence times of annotators of different institutions using the
    unpaired student-t test (parametric) as well its non-parametric equivalent,
    Wilcoxon rank-sum test or Mann-Whitney test (because it also works for
    discrete values). Performs significance tests for institutions AND groups
    of institutions.
    IMPORTANT: that if "later" is available, 5.H0-8.H0 are computed
    and it's assumed that MD doesn't include data from MD_LATER!

    Null hypotheses to be tested:
    1) Between institutions:
        1. H0: median confidence times in MD are drawn from the
        same distribution as median confidence times of SU

    2) between groups of institutions
        1. H0: median confidence times in MD (S) are drawn from the
        same distribution as median confidence times of SU (S)
        2. H0: median confidence times in MD (M) are drawn from the
        same distribution as median confidence times of SU (M)
        3. H0: median confidence times in MD (L) are drawn from the
        same distribution as median confidence times of SU (L)

    Info about which test to perform in what situation:
    http://stats.stackexchange.com/questions/121852/how-to-choose-between-t-test-or-non-parametric-test-e-g-wilcoxon-in-small-sampl
    http://stats.stackexchange.com/questions/2248/how-to-test-group-differences-on-a-five-point-variable
    http://blog.minitab.com/blog/adventures-in-statistics-2/best-way-to-analyze-likert-item-data:-two-sample-t-test-versus-mann-whitney

    Parameters
    ----------
    labels: dic - contains as values for the institutions ("md", "su", "later")
    a dict for each group ("S", "M", "L") with a list of median confidence
    times as a value.
    stat_path: str - path where infsci2017_results should be stored.

    """
    # Create median confidence times per institution as they are available
    # only per group per institution
    md_times = []
    for group in labels["md"]:
        md_times.extend(labels["md"][group])
    su_times = []
    for group in labels["su"]:
        su_times.extend(labels["su"][group])
    later_times = []
    if "later" in labels:
        for group in labels["later"]:
            later_times.extend(labels["later"][group])
    with open(stat_path, "wb") as f:
        # ##########################################################
        # Mann-Whitney test: doesn't assume continuous distribution
        # (non-parametric)
        # ##########################################################
        # 1.1. H0
        u, p = scipy.stats.mannwhitneyu(md_times, su_times, use_continuity=True,
                                        alternative="two-sided")
        f.write("Mann-Whitney (institution median conf times SU vs. MD)\n")
        f.write("----------------------------------------------------\n")
        write_significance(f, u, p)
        # 2.1. H0
        u, p = scipy.stats.mannwhitneyu(labels["md"]["S"], labels["su"]["S"],
                                        use_continuity=True,
                                        alternative="two-sided")
        f.write("Mann-Whitney (S) (group median conf times SU vs. MD)\n")
        f.write("----------------------------------------------------\n")
        write_significance(f, u, p)
        # 2.2. H0
        # If M is contained in the data (which doesn't have to be the case)
        if "M" in labels["md"] and "M" in labels["su"]:
            u, p = scipy.stats.mannwhitneyu(labels["md"]["M"],
                                            labels["su"]["M"],
                                            use_continuity=True,
                                            alternative="two-sided")
            f.write("Mann-Whitney (M) (group median conf times SU vs. MD)\n")
            f.write("----------------------------------------------------\n")
            write_significance(f, u, p)
        # 2.3. H0
        # If L is contained in the data (which doesn't have to be the case)
        if "L" in labels["md"] and "L" in labels["su"]:
            u, p = scipy.stats.mannwhitneyu(labels["md"]["L"],
                                            labels["su"]["L"],
                                            use_continuity=True,
                                            alternative="two-sided")
            f.write("Mann-Whitney (L) (group median conf times SU vs. MD)\n")
            f.write("----------------------------------------------------\n")
            write_significance(f, u, p)

        # ##########################################################
        # Wilcoxon ranked sum test: assumes continuous distribution
        # (but discrete should also work) (non-parametric)
        # ##########################################################
        # 1.1. H0
        u, p = scipy.stats.ranksums(md_times, su_times)
        f.write("Wilcoxon-ranked-sum (institution median conf times SU vs. "
                "MD)\n")
        f.write("----------------------------------------------------\n")
        write_significance(f, u, p)
        # 2.1. H0
        u, p = scipy.stats.ranksums(labels["md"]["S"], labels["su"]["S"])
        f.write("Wilcoxon-ranked-sum (S) (group median conf times SU vs. MD)\n")
        f.write("----------------------------------------------------\n")
        write_significance(f, u, p)
        # 2.2. H0
        # If M is contained in the data (which doesn't have to be the case)
        if "M" in labels["md"] and "M" in labels["su"]:
            u, p = scipy.stats.ranksums(labels["md"]["M"], labels["su"]["M"])
            f.write("Wilcoxon-ranked-sum (M) (group median conf times SU vs."
                    " MD)\n")
            f.write("----------------------------------------------------\n")
            write_significance(f, u, p)
        # 2.3. H0
        # If L is contained in the data (which doesn't have to be the case)
        if "L" in labels["md"] and "L" in labels["su"]:
            u, p = scipy.stats.ranksums(labels["md"]["L"], labels["su"]["L"])
            f.write("Wilcoxon-ranked-sum (L) (group median conf times SU vs. "
                    "MD)\n")
            f.write("----------------------------------------------------\n")
            write_significance(f, u, p)

        # ##########################################################
        # Unpaired student t-test: assumes normal distribution and continuous
        # distribution (parametric); use Baum-Welch which assumes different
        # variances of the data to be tested
        # ##########################################################
        # 1.1. H0
        u, p = scipy.stats.ttest_ind(md_times, su_times, equal_var=False)
        f.write("Baum-Welch (institution median conf times SU vs. "
                "MD)\n")
        f.write("----------------------------------------------------\n")
        write_significance(f, u, p)
        if len(later_times) > 0:
            # 1.2. H0
            u, p = scipy.stats.ttest_ind(md_times, later_times, equal_var=False)
            f.write("Baum-Welch (institution median conf times LATER "
                    "vs. MD)\n")
            f.write("----------------------------------------------------\n")
            write_significance(f, u, p)
        # 2.1. H0
        u, p = scipy.stats.ttest_ind(labels["md"]["S"], labels["su"]["S"],
                                     equal_var=False)
        f.write("Baum-Welch (S) (group median conf times SU vs. MD)\n")
        f.write("----------------------------------------------------\n")
        write_significance(f, u, p)
        # 2.2. H0
        # If M is contained in the data (which doesn't have to be the case)
        if "M" in labels["md"] and "M" in labels["su"]:
            u, p = scipy.stats.ttest_ind(labels["md"]["M"], labels["su"]["M"],
                                         equal_var=False)
            f.write("Baum-Welch (M) (group median conf times SU vs. MD)\n")
            f.write("----------------------------------------------------\n")
            write_significance(f, u, p)
        # 2.3. H0
        # If L is contained in the data (which doesn't have to be the case)
        if "L" in labels["md"] and "L" in labels["su"]:
            u, p = scipy.stats.ttest_ind(labels["md"]["L"], labels["su"]["L"],
                                         equal_var=False)
            f.write("Baum-Welch (L) (group median conf times SU vs. MD)\n")
            f.write("----------------------------------------------------\n")
            write_significance(f, u, p)


def write_significance(f, stat, p):
    """
    Write formatted significance infsci2017_results (statistic and p-value) given
    an open file handle. Also adds ** (or *) to the p-value if the p-value is
    significant on the 0.01 (or 0.05) significance level.

    Parameters
    ----------
    f: file handle - file in which the line should be written.
    stat: float - statistic of significance test for which a p-value was
    computed.
    p: float - p-value.

    """
    if p < 0.01:
        f.write("t: {:.8f}  p: {:.8f}**\n\n".format(stat, p))
    elif 0.01 <= p < 0.05:
        f.write("t: {:.8f}  p: {:.8f}*\n\n".format(stat, p))
    else:
        f.write("t: {:.8f}  p: {:.8f}\n\n".format(stat, p))


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    # Name of the collection in each DB holding annotator data
    ANNO_COLL_NAME = "user"
    # Name of the collection in each DB holding tweet data
    TWEET_COLL_NAME = "tweets"
    # Directory in which figures will be stored
    STAT_DIR = os.path.join(base_dir, "results", "stats",
                            "median_confidence_times")
    FIG_DIR = os.path.join(base_dir, "results", "figures",
                           "median_confidence_times")

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

    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    if not os.path.exists(STAT_DIR):
        os.makedirs(STAT_DIR)

    # Raw
    analyze_distribution(md_tweets, md_annos, su_tweets, su_annos, FIG_DIR,
                         STAT_DIR, cleaned=False)

    # Cleaned
    analyze_distribution(md_tweets, md_annos, su_tweets, su_annos, FIG_DIR,
                         STAT_DIR, cleaned=True)
