"""
Plots the confidence label distributions per
- group in an institution
- institution
Likewise, computes the significance level for these 2 possibilities.
"""

import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import pseudo_db

# For Information sciences
FONTSIZE = 12
plt.rcParams.update({'font.size': FONTSIZE})


def analyze_distribution(xlabels, fig_dst, stat_dst, md_tw_p, md_a_p, su_tw_p,
                         su_a_p, cleaned=False):
    """
    Plots label distribution by
    a) institution
    b) group per institution.
    Also computes the significance of a) and b) with and without annotators from
    group L (because only a few annotators were part of that
    # group, meaning the infsci2017_results could get biased as 350 tweets were labeled
    # only by 1 (for MD) or 3 (for SU) annotators).

    Parameters
    ----------
    xlabels: List of str - list of labels that should be displayed. Must be the
    same names as in the DB as they are used as keys to retrieve the values.
    fig_dst: str - directory in which the plot will be stored.
    stat_dst: str - directory in which the statistics will be stored.
    md_tw_p: str - path to MD tweets dataset in csv format.
    md_a_p: str - path to MD annotators dataset in csv format.
    su_tw_p: str - path to SU tweets dataset in csv format.
    su_a_p: str - path to SU annotators dataset in csv format.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing average
    annotation times.

    """
    agg_type = "raw"
    if cleaned:
        agg_type = "cleaned"

    # ##############################
    # Computations including L and M
    # ##############################
    # Get data grouped by
    # a) institution
    # b) group per institution (redundant as it's already considered above)
    md_insts, md_groups = get_conf_labels(md_tw_p, md_a_p, cleaned=cleaned,
                                          with_l=True, with_m=True)
    su_insts, su_groups = get_conf_labels(su_tw_p, su_a_p, cleaned=cleaned,
                                          with_l=True, with_m=True)
    insts = {"md": md_insts,
             "su": su_insts
             }
    groups = {
        "md": md_groups,
        "su": su_groups
    }

    # Plot confidence label distribution per institution
    fname = "{}_conf_label_distribution_with_l_with_m"\
        .format("_".join(l.lower() for l in xlabels))
    fpath_plot = os.path.join(fig_dst, fname + "_{}.pdf".format(agg_type))
    title = "{} confidence label distribution per institution".format(
        agg_type.title())
    _plot_distribution(insts, xlabels, title, fpath_plot)

    # Test statistical significance per institution
    name = "{}_conf_label_distribution_significance_with_l_with_m_{}"\
        .format("_".join(l.lower() for l in xlabels), agg_type)
    stat_path = os.path.join(stat_dst, name + ".txt")
    compute_significance_per_institution(insts, stat_path)

    # Plot confidence label distribution per group per institution
    fpath_plot = os.path.join(fig_dst, fname + "_per_group_{}.pdf"
                              .format(agg_type))
    title = "{} confidence label distribution per annotator group".format(
        agg_type.title())
    _plot_distribution_group(groups, xlabels, title, fpath_plot, with_l=True,
                             with_m=True)

    # Test statistical significance per group per institution
    name = \
        "{}_conf_label_distribution_significance_per_group_with_l_with_m_{}"\
        .format("_".join(l.lower() for l in xlabels), agg_type)
    stat_path = os.path.join(stat_dst, name + ".txt")
    compute_significance_per_group(groups, stat_path)

    # #########################################################################
    # Computations excluding L (only affects label distribution of institution)
    # #########################################################################
    # Get data grouped by
    # a) institution
    # b) group per institution (redundant as it's already considered above)
    md_insts, md_groups = get_conf_labels(md_tw_p, md_a_p, cleaned=cleaned,
                                          with_l=False, with_m=True)
    su_insts, su_groups = get_conf_labels(su_tw_p, su_a_p, cleaned=cleaned,
                                          with_l=False, with_m=True)
    insts = {"md": md_insts,
             "su": su_insts
             }
    groups = {
        "md": md_groups,
        "su": su_groups
    }

    # Plot confidence label distribution per institution
    fname = "{}_conf_label_distribution_without_l_with_m".format("_".join(
        l.lower() for l in xlabels))
    fpath_plot = os.path.join(fig_dst, fname + "_{}.pdf".format(agg_type))
    title = "{} confidence label distribution per institution".format(
        agg_type.title())
    _plot_distribution(insts, xlabels, title, fpath_plot)

    # Test statistical significance per institution
    name = "{}_conf_label_distribution_significance_without_l_with_m_{}"\
        .format("_".join(l.lower() for l in xlabels), agg_type)
    stat_path = os.path.join(stat_dst, name + ".txt")
    compute_significance_per_institution(insts, stat_path)

    # Store numbers of label distribution for paper
    p = os.path.join(stat_dst, "conf_label_distribution_numbers_without_l_"
                               "per_group_{}.txt".format(agg_type))
    with open(p, "wb") as f:
        f.write(str(groups))

    # Plot confidence label distribution per group per institution
    fpath_plot = os.path.join(fig_dst, fname + "_per_group_{}.pdf"
                              .format(agg_type))
    title = "{} confidence label distribution per annotator group".format(
        agg_type.title())
    _plot_distribution_group(groups, xlabels, title, fpath_plot, with_l=False,
                             with_m=True)

    # Test statistical significance per group per institution
    name = \
        "{}_conf_label_distribution_significance_per_group_without_l_with_m_" \
        "{}".format("_".join(l.lower() for l in xlabels), agg_type)
    stat_path = os.path.join(stat_dst, name + ".txt")
    compute_significance_per_group(groups, stat_path)


def get_conf_labels(tweet_path, anno_path, cleaned=False, with_l=False,
                    with_m=True):
    """
    Counts per annotator group in each institution how often each confidence
    labels was chosen.

    Parameters
    ----------
    tweet_path: str - path to tweet csv dataset.
    anno_path: str - path to annotator csv dataset.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing median
    annotation times.
    with_l: bool - True if group L should be considered in the results.
    with_m: bool - True if group M should be considered in the results.

    Returns
    -------
    Counter, dict.
    First Counter represents label counts in the institution ("md", "su").
    The next dict does the same,
    but for groups per  institution. Hence, keys are the group ("S", "M", "L")
    and value is again a Counter.

    """
    data = pseudo_db.Data(tweet_path, anno_path)
    # List of confidence labels assigned by annotators
    inst_annotations = []
    s_annotations = []
    m_annotations = []
    l_annotations = []
    # For each annotator
    for anno in data.annotators.all_annos():
        group = anno.get_group()
        # If we specified to exclude annotators from M or L, we skip this
        # loop. Otherwise, we extract all tweets of an annotator.
        if ((not with_l and group != "L") or with_l) \
                and ((not with_m and group != "M") or with_m):
            # For each tweet
            for t in anno.get_labeled_tweets():
                labels = t.get_anno_labels()
                conf_labels = t.get_conf_labels()
                # First level
                rel_label = labels[0]
                c1 = conf_labels[0]
                inst_annotations.append(c1)
                if group == "S":
                    s_annotations.append(c1)
                elif group == "M":
                    m_annotations.append(c1)
                else:
                    l_annotations.append(c1)
                # Discard remaining labels if annotator chose "Irrelevant"
                # Consider other sets of labels iff either the cleaned
                # dataset should be created and the label is "relevant" OR
                # the raw dataset should be used.
                if (cleaned and rel_label != "Irrelevant") or not cleaned:
                    l2 = labels[1]
                    # Second level
                    c2 = conf_labels[1]
                    inst_annotations.append(c2)
                    if group == "S":
                        s_annotations.append(c2)
                    elif group == "M":
                        m_annotations.append(c2)
                    else:
                        l_annotations.append(c2)
                    # Annotator labeled the 3rd set of labels as well
                    if l2 == pseudo_db.GROUND_TRUTH["Non-factual"]:
                        # Third level
                        c3 = conf_labels[2]
                        inst_annotations.append(c3)
                        if group == "S":
                            s_annotations.append(c3)
                        elif group == "M":
                            m_annotations.append(c3)
                        else:
                            l_annotations.append(c3)
    insts = Counter(inst_annotations)
    groups = {
        "S": Counter(s_annotations),
        "M": Counter(m_annotations),
        "L": Counter(l_annotations)
    }
    return insts, groups


def _plot_distribution(counters, xlabels, title, fpath):
    """
    Plots a bar chart given some counted labels.

    Parameters
    ----------
    counters: dict of collection.Counter - key is institution ("md", "su",
    "later") and value is a counter holding class label.
    counts for that institution.
    xlabels: List of str - list of labels for x-axis.
    displayed in the legend. Has the same order as <counters>.
    title: str - title of the plot.
    fpath: str - path where the plot is stored.

    """
    INSTITUTIONS = len(counters)
    print "#Institutions", INSTITUTIONS
    # Number of labels
    num_items = len(xlabels)
    # Color for each annotator group S, M, L
    COLORS = ["dodgerblue", "darkorange", "green"]
    # Label names in legend
    legend_labels = []
    y = []
    # Bar graphs expect a total width of "1.0" per group
    # Thus, you should make the sum of the two margins
    # plus the sum of the width for each entry equal 1.0.
    # One way of doing that is shown below. You can make
    # The margins smaller if they're still too big.
    # See
    # http://stackoverflow.com/questions/11597785/setting-spacing-between-grouped-bar-plots-in-matplotlib
    # to calculate gap in between bars; if gap isn't large enough, increase
    # <margin
    margin = 0.15
    width = (1.-3.*margin) / num_items
    ind = np.arange(num_items)
    # Build a <INSTITUTIONS>x<num_items> matrix
    for institution, counter in counters.iteritems():
        y_tmp = []
        # Get counts of the institution for the desired labels and normalize
        # counts (i.e. convert them to percentage)
        for label in xlabels:
            # Compute total number of labels because we want to display %
            # on y-axis, but consider only labels that are displayed in this
            # plot
            total = sum(counter[k] for k, v in counter.iteritems() if k in
                        xlabels)
            # print "TOTAL:", total
            print "institution:", institution
            # total = sum(counter.values())
            print "TOTAL:", total
            prcnt = 1.0 * counter[label] / total * 100
            y_tmp.append(prcnt)
        y.append(y_tmp)
        legend_labels.append(institution.upper())

    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)

    # Plot bars by displaying each annotator group after the other
    # Initially, each class adds 0 to its y-value as we start from the bottom
    # We have in total <Institution> graphs per label and each row represents
    # the percentages of the institutions for a different label
    bottom = np.zeros(shape=num_items,)
    # See
    # http://stackoverflow.com/questions/11597785/setting-spacing-between-grouped-bar-plots-in-matplotlib
    # to calculate gap in between bars; if gap isn't large enough, increase
    # <margin>
    for idx, y_ in enumerate(y):
        x = ind + margin + (idx * width)
        # If it's a stack, i.e. not the one at the bottom, always add the bottom
        ax.bar(x, y_, width, label=legend_labels[idx], color=COLORS[idx])
        # Add current y-values to the bottom, so that next bar starts directly
        # above
        bottom += y_
    # plt.title(title)
    # Set title position
    ttl = ax.title
    ttl.set_position([.5, 1.05])
    # Set labels for ticks
    ax.set_xticks(ind+0.5)
    # ax.set_xticklabels(xlabels)
    ax.set_xticklabels(xlabels, rotation=45)
    # y-axis from 0-100
    plt.yticks(np.arange(0, 110, 10))
    # Hide the right and top spines (lines)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    # Set labels of axes
    # ax.set_xlabel("Labels")
    ax.set_ylabel("Percentage")
    # Add a legend
    # ax.legend(loc="center left", shadow=True, bbox_to_anchor=(1, 0.5))
    ax.legend(loc="best", shadow=True, fontsize=FONTSIZE)
    plt.savefig(fpath, bbox_inches="tight", dpi=600)


def _plot_distribution_group(counters, xlabels, title, fpath, with_l, with_m):
    """
    Plots a stacked bar chart breaking the number of labels down to annotator
    groups.

    Parameters
    ----------
    counters: dict of collection.Counter - key is institution ("md", "su",
    "later") and value is a counter holding class label
    counts for that institution.
    xlabels: List of str - list of labels for x-axis.
    displayed in the legend. Has the same order as <counters>.
    title: str - title of the plot.
    fpath: str - path where the plot is stored.
    with_l: bool - True if group L should be considered in the esults.
    with_m: bool - True if group M should be considered in the results.

    """
    print xlabels
    INSTITUTIONS = len(counters)
    print "#Institutions", INSTITUTIONS
    # Number of labels
    num_items = len(xlabels)
    # Color for each annotator group S, M, L
    COLORS = ["dodgerblue", "darkorange", "green"]
    # Number of annotator groups per institution
    GROUPS = len(counters["md"])
    print "GROUPS", GROUPS
    # Label names in legend
    legend_labels = []
    y = []
    # Bar graphs expect a total width of "1.0" per group
    # Thus, you should make the sum of the two margins
    # plus the sum of the width for each entry equal 1.0.
    # One way of doing that is shown below. You can make
    # The margins smaller if they're still too big.
    # See
    # http://stackoverflow.com/questions/11597785/setting-spacing-between-grouped-bar-plots-in-matplotlib
    # to calculate gap in between bars; if gap isn't large enough, increase
    # <margin
    margin = 0.15
    width = (1.-3.*margin) / num_items
    ind = np.arange(num_items)
    # Get each label to be displayed over bars once. It's only used to
    # learn in which order the data is plotted
    tmp_bar_labels = []
    group_labels = []
    # Build a <INSTITUTIONS>x<num_items> matrix, where a column contains all
    # counts for an institution.
    for institution, groups in counters.iteritems():
        print "Institution", institution
        # Stores raw counts for institution
        raws = []
        # For each annotator group separately
        for idx, (group, counter) in enumerate(groups.iteritems()):
            print counter
            print "Group", idx, group
            # Add legend label if it's not included yet
            if group not in group_labels:
                # Add "M" and "L" only if M and LL should be considered
                if ((not group == "L" and not with_l) or with_l) \
                        and ((not group == "M" and not with_m) or with_m):
                    group_labels.append(group.upper())
            # Get counts of the institution for the desired labels and normalize
            # counts (i.e. convert them to percentage)
            tmp = []
            for label in xlabels:
                # print "Label", label
                # Compute total number of labels because we want to display %
                # on y-axis, but consider only labels that are displayed in this
                # plot
                # total = sum(counter[k] for k, v in counter.iteritems() if k in
                #             xlabels)
                # print "TOTAL:", total
                # if total > 0:
                    # prcnt = 1.0 * counter[label] / total * 100
                    # colors.append(COLORS[group])
                tmp.append(counter[label])
            raws.append(tmp)
            legend_labels.append(institution.upper())

            # Compute total only if all values per institution are available
            # Skip this loop in the first iteration
            if idx > 0 and idx % (GROUPS-1) == 0:
                total = 0
                print raws
                # <raws>: list of lists. Inner lists contain <xlabels> elements,
                # i.e. 1 count per label
                for g in raws:
                    total += sum(g)
                # print "TOTAL for group:", total
                # Convert raw counts into percentages
                for g in raws:
                    y_tmp = []
                    for count in g:
                        if total > 0:
                            prcnt = 1.0 * count / total * 100
                            y_tmp.append(prcnt)
                    y.append(y_tmp)
                    # print "normalized:", y_tmp
                raws = []
                # print "raws was cleaned:", raws
                # print "Y:", y
        tmp_bar_labels.append(institution.upper())
    # print "Y:", y

    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)

    # Plot bars by displaying each annotator group after the other
    # Initially, each class adds 0 to its y-value as we start from the bottom
    # We have in total <Institution> graphs per label and each row represents
    # the percentages of the institutions for a different label
    bottom = np.zeros(shape=num_items,)

    bar_labels = tmp_bar_labels * num_items
    # Color index
    col_idx = 0
    idx = 0
    # i-th bar
    i = 0
    bottoms = []
    # X-values of bars
    xbars = []
    # List containing some bars representing S, M, L (in total each group is
    # represented in here exactly once)
    legend_handles = []
    # Calculate x's and y's for plotting
    for y_ in y:
        # Initialize bottom for each institution separately
        if idx % GROUPS == 0:
            # Store <bottom> because we need to add text on top of each bar
            # later; append each y-value separetely
            # if idx > 0:
            #     for bot in bottom:
            #         print "bottom y to add", bot
            #         bar_heights.append(bot)
            # Now empty it for the next institution
            bottom = np.zeros(shape=num_items,)
            # Results must be stacked, so always go through the same <xlabel>
            # x-coords for the same institution
            x = ind + margin + (i * width)
            xbars.append(x)
            i += 1
            # print "idx", idx

        if len(y_) > 0:
            # print "use color", COLORS[col_idx]
            # print "x", x
            # print "now plot", y_
            p = ax.bar(x, y_, width, bottom=bottom,
                       color=COLORS[col_idx % len(COLORS)])
            # Add only the first GROUP bars to the legend as the same ones
            # are used for the other institutions
            if idx < GROUPS:
                legend_handles.append(p)
            # Add current y-values to the bottom, so that next bar starts
            # directly above
            bottom += y_
        col_idx += 1
        idx += 1
        # Store height of highest rectangle in the plot (at idx=0 everything is
        # empty)
        if idx % GROUPS == 0:
            # print "store bottom: {} for institution {}".format(bottom, idx)
            bottoms.append(bottom)
    # print "bottom for ys:", bottoms

    # Rearrange order of bar heights. Now they are [[a,b], [c,d], [e,f]], i.e.
    # 2 labels are used and for each group ([a,b] is 1 institution) there
    # exists a bar. We want [a, c, e, b, d, f]. Note that a-f are the heights
    # of the respective bars.
    bar_heights = []
    # Use i-th height of each array first
    for i in xrange(num_items):
        for ii, inst in enumerate(bottoms):
            bar_heights.append(bottoms[ii][i])
    # print "ordered heights:", bar_heights

    # Convert x-values of bars into correct format. Same problem as with
    # <bar_heights> above.
    # Use i-th x-value of each array first
    bar_starts = []
    for i in xrange(num_items):
        for ii, inst in enumerate(xbars):
            bar_starts.append(xbars[ii][i])
    # print "ordered x:", bar_starts
    # plt.title(title)
    # Set title position
    ttl = ax.title
    ttl.set_position([.5, 1.05])
    # Set labels for ticks
    ax.set_xticks(ind+0.5)
    # ax.set_xticklabels(xlabels)
    ax.set_xticklabels(xlabels, rotation=45)
    # y-axis from 0-100
    plt.yticks(np.arange(0, 110, 10))
    # Hide the right and top spines (lines)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    # Set labels of axes
    # ax.set_xlabel("Labels")
    ax.set_ylabel("Percentage")

    # Add institution name above each bar in the plot
    # http://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart
    rects = ax.patches
    for idx, (rect, label) in enumerate(zip(rects, bar_labels)):
        height = bar_heights[idx]
        ax.text(bar_starts[idx] + rect.get_width()/2, height + 5, label,
                ha='center', va='bottom')

    # Add a legend manually
    # http://stackoverflow.com/questions/16826711/is-it-possible-to-add-a-string-as-a-legend-item-in-matplotlib
    ax.legend(legend_handles, group_labels, shadow=True, loc="best",
              fontsize=FONTSIZE)
    plt.savefig(fpath, bbox_inches="tight", dpi=600)


def compute_significance_per_institution(labels, stat_path):
    """
    Computes if there is a statistically significant difference in the label
    assignment behavior of annotators of different institutions using the
    unpaired student-t test (parametric) as well its non-parametric equivalent,
    Wilcoxon rank-sum test or Mann-Whitney test (because it also works for
    discrete values)

    Null hypotheses to be tested:
    //1. H0: MD High is drawn from same distribution as SU High -> can't be tested because all values are 2's
    //2. H0: MD Low is drawn from same distribution as SU Low -> can't be tested because all values are 1's
    3. H0: MD Low + MD High is drawn from same distribution as SU Low + SU High

    Info about which test to perform in what situation:
    http://stats.stackexchange.com/questions/121852/how-to-choose-between-t-test-or-non-parametric-test-e-g-wilcoxon-in-small-sampl
    http://stats.stackexchange.com/questions/2248/how-to-test-group-differences-on-a-five-point-variable
    http://blog.minitab.com/blog/adventures-in-statistics-2/best-way-to-analyze-likert-item-data:-two-sample-t-test-versus-mann-whitney

    Parameters
    ----------
    labels: dic - contains as values for the institutions ("md", "su") a Counter
    object that holds how often each label ("high", "low")  occurred.
    stat_path: str - path where infsci2017_results should be stored.

    """
    su_high = labels["su"]["High"]
    su_low = labels["su"]["Low"]
    md_high = labels["md"]["High"]
    md_low = labels["md"]["Low"]
    # Create sequences of 1's and 2's representing "low" and "high"
    # MD
    md_high = [2.] * md_high
    md_low = [1.] * md_low
    md = md_low + md_high
    # SU
    su_high = [2.] * su_high
    su_low = [1.] * su_low
    su = su_low + su_high

    with open(stat_path, "wb") as f:
        # ##########################################################
        # Mann-Whitney test: doesn't assume continuous distribution
        # (non-parametric)
        # ##########################################################
        # 1. H0
        # u, p = scipy.stats.mannwhitneyu(md_high, su_high, use_continuity=False)
        # f.write("Mann-Whitney (SU high vs. MD high)\n")
        # f.write("----------------------------------\n")
        # f.write("u: {:.8f}  p: {:.8f}\n\n".format(u, p))
        # # 2. H0
        # u, p = scipy.stats.mannwhitneyu(md_low, su_low, use_continuity=False)
        # f.write("Mann-Whitney (SU low vs. MD low)\n")
        # f.write("--------------------------------\n")
        # f.write("u: {:.8f}  p: {:.8f}\n\n".format(u, p))
        # 3. H0
        u, p = scipy.stats.mannwhitneyu(md, su, use_continuity=False,
                                        alternative="two-sided")
        f.write("Mann-Whitney (SU low + SU high vs. MD low + MD high)\n")
        f.write("----------------------------------------------------\n")
        write_significance(f, u, p)

        # ##########################################################
        # Wilcoxon ranked sum test: assumes continuous distribution
        # (non-parametric)
        # ##########################################################
        # 1. H0
        # stat, p = scipy.stats.ranksums(md, su)
        # f.write("Wilcoxon ranked sum (SU high vs. MD high)\n")
        # f.write("-------------------\n")
        # f.write("statistics:{:.8f} p: {:.8f}\n\n".format(stat, p))
        # # 2. H0
        # stat, p = scipy.stats.ranksums(md, su)
        # f.write("Wilcoxon ranked sum (SU low vs. MD low)\n")
        # f.write("-------------------\n")
        # f.write("statistics:{:.8f} p: {:.8f}\n\n".format(stat, p))
        # 3. H0
        stat, p = scipy.stats.ranksums(md, su)
        f.write("Wilcoxon ranked sum (SU low + SU high vs. MD low + MD high)\n")
        f.write("-------------------\n")
        write_significance(f, stat, p)

        # ##########################################################
        # Unpaired student t-test: assumes normal distribution and continuous
        # distribution (parametric); use Baum-Welch which assumes different
        # variances of the data to be tested
        # ##########################################################
        # 1. H0
        # t, p = scipy.stats.ttest_ind(md_high, su_high, equal_var=False)
        # f.write("Baum-Welch (SU high vs. MD high)\n")
        # f.write("----------\n")
        # f.write("t: {:.8f}  p: {:.8f}\n\n".format(t, p))
        # # 2. H0
        # t, p = scipy.stats.ttest_ind(md_low, su_low, equal_var=False)
        # f.write("Baum-Welch (SU low vs. MD low)\n")
        # f.write("----------\n")
        # f.write("t: {:.8f}  p: {:.8f}\n\n".format(t, p))
        # 3. H0
        t, p = scipy.stats.ttest_ind(md, su, equal_var=False)
        f.write("Baum-Welch (SU low + SU high vs. MD low + MD high)\n")
        f.write("----------\n")
        write_significance(f, t, p)


def compute_significance_per_group(labels, stat_path):
    """
    Computes if there is a statistically significant difference in the label
    assignment behavior of annotators of groups of different institutions using
    the unpaired student-t test (parametric) as well its non-parametric
    equivalent, Wilcoxon rank-sum test or Mann-Whitney test (because it also
    works for discrete values)

    Null hypotheses to be tested:
    1. H0: MD Low + MD High S is drawn from same distribution as SU Low +
    SU High S
    2. H0: MD Low + MD High M is drawn from same distribution as SU Low +
    SU High M
    3. H0: MD Low + MD High L is drawn from same distribution as SU Low +
    SU High L

    Info about which test to perform in what situation:
    http://stats.stackexchange.com/questions/121852/how-to-choose-between-t-test-or-non-parametric-test-e-g-wilcoxon-in-small-sampl
    http://stats.stackexchange.com/questions/2248/how-to-test-group-differences-on-a-five-point-variable
    http://blog.minitab.com/blog/adventures-in-statistics-2/best-way-to-analyze-likert-item-data:-two-sample-t-test-versus-mann-whitney

    Parameters
    ----------
    labels: dic - contains as values for the institutions ("md", "su") per
    group ("S", "M", "L") a Counter object that holds how often each label
    ("high", "low")  occurred.
    stat_path: str - path where infsci2017_results should be stored.

    """
    with open(stat_path, "wb") as f:
        # Test 1. H0, 2. H0, 3. H0 in that order
        for group in ["S", "M", "L"]:
            su_high = labels["su"][group]["High"]
            su_low = labels["su"][group]["Low"]
            md_high = labels["md"][group]["High"]
            md_low = labels["md"][group]["Low"]
            # Create sequences of 1's and 2's representing "low" and "high"
            md = [1.] * md_low + [2.] * md_high
            su = [1.] * su_low + [2.] * su_high

            # ##########################################################
            # Mann-Whitney test: doesn't assume continuous distribution
            # (non-parametric)
            # ##########################################################
            u, p = scipy.stats.mannwhitneyu(md, su, use_continuity=False,
                                            alternative="two-sided")
            f.write("Mann-Whitney {} "
                    "(SU low + SU high vs. MD low + MD high)\n".format(group))
            f.write("-------------------------------------------------------\n")
            write_significance(f, u, p)

            # ##########################################################
            # Wilcoxon ranked sum test: assumes continuous distribution
            # (non-parametric)
            # ##########################################################
            stat, p = scipy.stats.ranksums(md, su)
            f.write("Wilcoxon ranked sum {} "
                    "(SU low + SU high vs. MD low + MD high)\n".format(group))
            f.write("-------------------------------------------------------\n")
            write_significance(f, stat, p)

            # ##########################################################
            # Unpaired student t-test: assumes normal distribution and continuous
            # distribution (parametric); use Baum-Welch which assumes different
            # variances of the data to be tested
            # ##########################################################
            t, p = scipy.stats.ttest_ind(md, su, equal_var=False)
            f.write("Baum-Welch {} "
                    "(SU low + SU high vs. MD low + MD high)\n".format(group))
            f.write("-------------------------------------------------------\n")
            write_significance(f, t, p)


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
    LABELS = ["High", "Low"]
    # Directory in which statistical tests will be stored
    STAT_DIR = os.path.join(base_dir, "results", "stats",
                            "conf_label_distribution")
    # Directory in which figures will be stored
    FIG_DIR = os.path.join(base_dir, "results", "figures",
                           "conf_label_distribution")

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
    analyze_distribution(LABELS, FIG_DIR, STAT_DIR, md_tweets, md_annos,
                         su_tweets, su_annos, cleaned=False)
    # Cleaned
    analyze_distribution(LABELS, FIG_DIR, STAT_DIR, md_tweets, md_annos,
                         su_tweets, su_annos, cleaned=True)
