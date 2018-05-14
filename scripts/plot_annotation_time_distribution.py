"""
Plots the distribution of the annotation times per group and institution.
This allows to check if they are approximately normally distributed.
Additionally, also plots if annotation times are transformed by 1/annotation
time. (If it's a linear transformation, plotting old values vs. transformed ones
yields a line. - not implemented yet)
Also plots Log_2 transformation which yields something similar to normal
distribution.
"""
import os
import math

import matplotlib.pyplot as plt

import pseudo_db


def plot_anno_time_distribution(fig_dst, md_tw_p, md_a_p, su_tw_p, su_a_p,
                                cleaned=False):
    """
    Plots the distrbution of all annotation times per:
    a) group per institution
    b) institution
    Also plots the distributions for a) and b) with transformed annotation times
    where each annotation time is replaced by its reciprocal, i.e. 1/annotation
    time.
    Also plots if the transformation is linear or not. If it's linear, plotting
    old values vs. transformed values yields a line.

    Parameters
    ----------
    fig_dst: str - directory in which the plot will be stored.
    md_tw_p: str - path to MD tweets dataset in csv format.
    md_a_p: str - path to MD annotators dataset in csv format.
    su_tw_p: str - path to SU tweets dataset in csv format.
    su_a_p: str - path to SU annotators dataset in csv format.
    cleaned: bool - True if the cleaned data is used as input.

    """
    agg_type = "raw"
    if cleaned:
        agg_type = "cleaned"

    su_inst, su_group = get_total_anno_times(su_tw_p, su_a_p, cleaned=cleaned,
                                             with_l=False)

    md_inst, md_group = get_total_anno_times(md_tw_p, md_a_p, cleaned=cleaned,
                                             with_l=False)

    inst_times = {
        "md": md_inst,
        "su": su_inst
    }

    group_times = {
        "md": md_group,
        "su": su_group
    }
    # inst_times, group_times = get_anno_times_from_dbs(
    #     DB_NAMES, SU_ALL, MD_ALL, tweet_coll_name=tweet_coll_name,
    #     anno_coll_name=anno_coll_name, cleaned=cleaned, with_l=False)

    # 1. Plot distributions per group and institution
    # 1a) plot annotation time distribution per group
    for inst in group_times:
        for group in group_times[inst]:
            print inst, group
            print "#ys", len(group_times[inst][group])
            fname = "{}_{}_anno_times_distribution_{}.png"\
                .format(inst, group.lower(), agg_type)
            dst = os.path.join(fig_dst, fname)
            _plot(group_times[inst][group], dst)

    # 1b) plot annotation time distribution per institution
    for inst in inst_times:
        fname = "{}_anno_times_distribution_{}.png"\
                .format(inst, agg_type)
        dst = os.path.join(fig_dst, fname)
        _plot(inst_times[inst], dst)

    # 2. Plot distributions per group and institution with transformed
    # annotation times (anno_time_new = 1/anno_time_old
    # 2a) plot annotation time distribution per group
    for inst in group_times:
        for group in group_times[inst]:
            transformed = [1.0 / t for t in group_times[inst][group]]
            fname = "{}_{}_anno_times_distribution_reciprocal_{}.png"\
                .format(inst, group.lower(), agg_type)
            dst = os.path.join(fig_dst, fname)
            _plot(transformed, dst)

    # 2b) plot annotation time distribution per institution
    for inst in inst_times:
        fname = "{}_anno_times_distribution_reciprocal_{}.png"\
                .format(inst, agg_type)
        dst = os.path.join(fig_dst, fname)
        transformed = [1.0 / t for t in inst_times[inst]]
        _plot(transformed, dst)

    # 3. Plot distributions per group and institution with transformed
    # annotation times (anno_time_new = log_2(anno_time_old)
    # 3a) plot annotation time distribution per group
    for inst in group_times:
        for group in group_times[inst]:
            transformed = [math.log(t if t > 0 else 1) for t in
                           group_times[inst][group]]
            fname = "{}_{}_anno_times_distribution_log2_{}.png"\
                .format(inst, group.lower(), agg_type)
            dst = os.path.join(fig_dst, fname)
            _plot(transformed, dst)

    # 3b) plot annotation time distribution per institution
    for inst in inst_times:
        fname = "{}_anno_times_distribution_log2_{}.png"\
                .format(inst, agg_type)
        dst = os.path.join(fig_dst, fname)
        transformed = [math.log(t if t > 0 else 1) for t in inst_times[inst]]
        _plot(transformed, dst)


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
    List, dict.
    Annotation times per tweet for each annotator (= inner list) in the whole
    institution.
    [[total times of annotator1], [total times of annotato2], ...]
    Annotation times grouped annotator group.
    {
        "S": [[total times by annotator1], [total times by annotator2], ...],
        "M": [[...], [...], ...],
        "L": [[...], [...], ...]
    }

    """
    insts = []
    # List of all annotation times per group
    groups = {
            "S": [],
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
            # First level
            rel_label = labels[0]
            anno_times.append(a_times[0])
            # Discard remaining labels if annotator chose "Irrelevant"
            # Consider other sets of labels iff either the cleaned
            # dataset should be created and the label is "relevant" OR
            # the raw dataset should be used.
            if (cleaned and rel_label != "Irrelevant") or not cleaned:
                # Second level
                l2 = labels[1]
                anno_times.append(a_times[1])
                # Annotator labeled the 3rd set of labels as well
                if l2 == pseudo_db.GROUND_TRUTH["Non-factual"]:
                    # Third level
                    anno_times.append(a_times[2])

            insts.extend(anno_times)
            groups[group].extend(anno_times)
    if not with_l:
        del groups["L"]
    return insts, groups


def _plot(anno_times, dst):
    """
    Plots annotation times in histogram.

    Parameters
    ----------
    anno_times: list of float - list of annotation times to be binned.
    dst: str - path in which plot should be stored.

    """
    # Number of bins used in the histogram
    bins = int(round(math.sqrt(len(anno_times))))
    print "BINS:", bins
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    # x = range(max(anno_times))

    n, bins, _ = ax.hist(anno_times, bins=bins, normed=True,
                         histtype='stepfilled', alpha=0.2)
    # Hide the right and top spines (lines)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # Limits of axes
    x = ax.get_xlim()
    plt.xlim(0, x[1])
    # plt.ylim(0, max(max(n), max(y_fit)))
    # Set labels of axes
    ax.set_xlabel("Annotation time in s")
    ax.set_ylabel("Probability")
    # ax.set_ylabel("Probability")
    # Add legend outside of plot
    legend = ax.legend(loc="best", shadow=True, bbox_to_anchor=(0.5, 1.5))
    plt.savefig(dst, bbox_inches='tight')
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

    # Directory in which figures will be stored
    FIG_DIR = os.path.join(base_dir, "results", "figures",
                           "anno_time_distribution")
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    # Raw
    plot_anno_time_distribution(FIG_DIR, md_tweets, md_annos, su_tweets,
                                su_annos, cleaned=False)
    # Cleaned
    plot_anno_time_distribution(FIG_DIR, md_tweets, md_annos, su_tweets,
                                su_annos, cleaned=True)
