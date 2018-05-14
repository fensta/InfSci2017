"""
Plotting is done in this script
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import pseudo_db


def plot_median_anno_time_per_institution(tweet_path, anno_path, dst,
                                          cleaned=False):
    """
    Aggregates all median annotation times of an institution into a single
    plot grouped by annotation groups (encoded by using a different color per
    group).
    The file name is of the form:
    <institution>_median_annotation_time_distribution.png.

    Parameters
    ----------
    tweet_path: str - path to tweet csv dataset.
    anno_path: str - path to annotator csv dataset.
    dst: str - directory in which the plot will be stored.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing average
    annotation times.

    """
    # Find out for which institution this plot is - su or md
    data = pseudo_db.Data(tweet_path, anno_path)
    institution = data.annotators.get_ith_annotator(0).get_dataset_name()
    dataset_type = "raw"
    if cleaned:
        dataset_type = "cleaned"
    # Average annotation time per annotator on the y-axis (per annotator group)
    median_times_s = []
    median_times_m = []
    median_times_l = []
    # Each annotator is represented on the x-axis (per annotator group)
    annos_s = []
    annos_m = []
    annos_l = []
    # For each annotator
    for idx, anno in enumerate(data.annotators.all_annos()):
        # Overall annotation times for an annotator per tweet in s
        anno_times = []
        username = anno.get_name()
        group = anno.get_group()
        print "Collect tweets for annotator '{}'".format(username)
        # For each tweet
        for t in anno.get_labeled_tweets():
            labels = t.get_anno_labels()
            a_times = t.get_anno_times()
            # First level
            rel_label = labels[0]
            a1 = a_times[0]
            # Second and third level might have not been assigned
            a2 = pseudo_db.ZERO
            a3 = pseudo_db.ZERO
            # Discard remaining labels if annotator chose "Irrelevant"
            # Consider other sets of labels only iff either the cleaned
            # dataset should be created and the label is "relevant" OR
            # the raw dataset should be used.
            if (cleaned and rel_label != "Irrelevant") or not cleaned:
                # Second level
                l2 = labels[1]
                a2 = a_times[1]
                # Annotator labeled the 3rd set of labels as well
                if l2 == pseudo_db.GROUND_TRUTH["Non-factual"]:
                    # Third level
                    a3 = a_times[2]
            ls = [a1, a2, a3]
            # Sum up relative times
            total = sum(ls)
            anno_times.append(total)
        times = np.array(anno_times, dtype=float)
        median_time = np.nanmedian(times, axis=0).T
        if group == "S":
            median_times_s.append(median_time)
            annos_s.append(username)
        elif group == "M":
            median_times_m.append(median_time)
            annos_m.append(username)
        else:
            median_times_l.append(median_time)
            annos_l.append(username)
    # Plot average times per annotator group
    x_s = [i+1 for i in range(len(annos_s))]
    # Don't forget to add the offset of S
    x_m = [i+1 + len(annos_s) for i in range(len(annos_m))]
    # Don't forget to add the offset of S+M
    x_l = [i+1 + len(annos_s) + len(annos_m) for i in range(len(annos_l))]
    # print "x_s:", x_s
    # print "x_m:", x_m
    # print "x_l:", x_l
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)

    # Set the size of each data point (=annotator)
    # S
    s = [10 for n in range(len(annos_s))]
    # Use different colors to encode the annotator group of each participant
    ax.scatter(x_s, median_times_s, color="green", s=s, label="Annotator group S")

    # M
    s = [10 for n in range(len(annos_m))]
    # Use different colors to encode the annotator group of each participant
    ax.scatter(x_m, median_times_m, color="black", s=s,
               label="Annotator group M")
    # L
    s = [10 for n in range(len(annos_l))]
    # Use different colors to encode the annotator group of each participant
    ax.scatter(x_l, median_times_l, color="red", s=s,
               label="Annotator group L")

    # Set range of axes
    if len(x_l) > 0:
        plt.xlim(0, x_l[-1] + 1)
    else:
        plt.xlim(0, x_s[-1] + 1)
    # Find the maximum median annotation time over all annotator groups
    max_time = max(max(median_times_s), max(median_times_m),
                   max(median_times_l))
    print "times"
    print median_times_s
    print median_times_m
    print median_times_l
    plt.ylim(0, max_time + 3)
    # Hide the right and top spines (lines)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # Set labels of axes
    ax.set_xlabel("Annotator ID")
    ax.set_ylabel("Median annotation time in s")
    # Add legend outside of plot
    legend = ax.legend(loc='upper center', shadow=True,
                       bbox_to_anchor=(0.5, 1.5))
    fname = "{}_median_annotation_time_distribution_{}.png"\
        .format(institution, dataset_type)
    # Save plot
    plt.savefig(os.path.join(dst, fname), bbox_inches='tight')


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
                           "median_annotation_time_su_all")
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)

    # SU
    # Raw
    plot_median_anno_time_per_institution(su_tweets, su_annos, FIG_DIR,
                                          cleaned=False)
    # Cleaned
    plot_median_anno_time_per_institution(su_tweets, su_annos, FIG_DIR,
                                          cleaned=True)
    FIG_DIR = os.path.join(base_dir, "results", "figures",
                           "median_annotation_time_md_all")
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    # MD
    # Raw
    plot_median_anno_time_per_institution(md_tweets, md_annos, FIG_DIR,
                                          cleaned=False)
    # Cleaned
    plot_median_anno_time_per_institution(md_tweets, md_annos, FIG_DIR,
                                          cleaned=True)
