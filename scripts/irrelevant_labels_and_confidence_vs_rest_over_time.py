"""
Investigate confidence and annotation times of irrelevant vs. the other 2
labels.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from statsmodels import robust

import pseudo_db


# For Information sciences
FONTSIZE = 12
plt.rcParams.update({'font.size': FONTSIZE})


def plot_irrelevant_vs_rest_per_institution_and_anno(
     tweet_path, anno_path, dst, cleaned=False, k=0, color="red"):
    """
    Plots 5 bars: irrelevant annotation/confidence time, relevant
    annotation/confidence time, 2nd set of labels/confidence, 3rd set of labels/
    confidence times.
    Uses either average or median for the computation of these times.
    Plots the infsci2017_results per institution, and per annotator.

    Parameters
    ----------
    tweet_path: str - path to tweet csv dataset.
    anno_path: str - path to annotator csv dataset.
    dst: str - directory in which the plot will be stored.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing average
    annotation times
    k: int - number of tweets that should be discarded from the beginning due
    to the learning effect.
    color: str - color in matplotlib used to fill the bars.

    """
    # k actually represents the index and we start counting at 0, hence -1
    k -= 1

    data_type = "raw"
    if cleaned:
        data_type = "cleaned"

    # Annotation times of "irrelevant" tweets of all annotators
    irrel_times = []
    # Annotation times of "relevant" tweets of all annotators
    rel_times = []
    # Annotation times of 2nd set of labels of all annotators
    second_times = []
    # Annotation times of 3rd set of labels of all annotators
    third_times = []

    # Same as above, but for confidence times
    irrel_conf_times = []
    rel_conf_times = []
    second_conf_times = []
    third_conf_times = []

    data = pseudo_db.Data(tweet_path, anno_path)
    # For each annotator
    for anno in data.annotators.all_annos():
        name = anno.get_name()
        group = anno.get_group()
        # print "Collect tweets for annotator '{}'".format(username)
        # Number of tweets to skip in the beginning of each annotator
        skipped = 0
        # Same datastructures as above, but only for a single annotator
        anno_irrel_times = []
        anno_rel_times = []
        anno_second_times = []
        anno_third_times = []
        anno_irrel_conf_times = []
        anno_rel_conf_times = []
        anno_second_conf_times = []
        anno_third_conf_times = []
        # For each tweet
        for idx, t in enumerate(anno.get_labeled_tweets()):
            # After skipping the first k tweets, continue as usual computing
            # confidence times
            if skipped >= k:
                labels = t.get_anno_labels()
                anno_times = t.get_anno_times()
                conf_times = t.get_conf_times()
                # True if the 3rd set of labels was assigned by annotator
                has_opinion = False
                # True, if the tweet isn't "irrelevant"
                has_more = False
                # First level
                rel_label = labels[0]
                a1 = anno_times[0]
                c1 = conf_times[0]
                # 2nd and 3rd level might not have been assigned at all
                a2 = pseudo_db.ZERO
                c2 = pseudo_db.ZERO
                a3 = pseudo_db.ZERO
                c3 = pseudo_db.ZERO
                # Discard remaining labels if annotator chose "Irrelevant"
                # Consider other sets of labels only iff either the cleaned
                # dataset should be created and the label is "relevant" OR
                # the raw dataset should be used.
                if (cleaned and rel_label != "Irrelevant") or not cleaned:
                    has_more = True
                    # Second level
                    l2 = labels[1]
                    a2 = anno_times[1]
                    c2 = conf_times[1]
                    # Annotator labeled the 3rd set of labels as well
                    if l2 == pseudo_db.GROUND_TRUTH["Non-factual"]:
                        # Third level
                        a3 = anno_times[2]
                        c3 = conf_times[2]
                        has_opinion = True
                # If "irrelevant" was assigned and we want to discard other
                # labels, add only times for 1st set of labels
                if cleaned and rel_label == "Irrelevant":
                    anno_irrel_times.append(a1)
                    anno_irrel_conf_times.append(c1)
                # Tweet is "relevant" and also has times for 2nd and 3rd
                # set of labels
                elif cleaned and has_more and has_opinion:
                    anno_rel_times.append(a1)
                    anno_rel_conf_times.append(c1)
                    anno_second_times.append(a2)
                    anno_second_conf_times.append(c2)
                    anno_third_times.append(a3)
                    anno_third_conf_times.append(c3)
                # Tweet is "relevant" and has 2nd set of labels
                elif cleaned and has_more and not has_opinion:
                    anno_rel_times.append(a1)
                    anno_rel_conf_times.append(c1)
                    anno_second_times.append(a2)
                    anno_second_conf_times.append(c2)
                # Tweet is "irrelevant" or "relevant", but we consider all
                # labels
                else:
                    # Relevant or irrelevant tweet?
                    if rel_label == "Irrelevant":
                        anno_irrel_times.append(a1)
                        anno_irrel_conf_times.append(c1)
                    else:
                        anno_rel_times.append(a1)
                        anno_rel_conf_times.append(c1)
                    anno_second_times.append(a2)
                    anno_second_conf_times.append(c2)
                    # If 3rd set of labels was assigned
                    if has_opinion:
                        anno_third_times.append(a3)
                        anno_third_conf_times.append(c3)
            else:
                # We just skipped another tweet
                skipped += 1
        # Add data from a single anno to global lists
        irrel_times.extend(anno_irrel_times)
        rel_times.extend(anno_rel_times)
        second_times.extend(anno_second_times)
        third_times.extend(anno_third_times)
        irrel_conf_times.extend(anno_irrel_conf_times)
        rel_conf_times.extend(anno_rel_conf_times)
        second_conf_times.extend(anno_second_conf_times)
        third_conf_times.extend(anno_third_conf_times)

        # Plot annotation times per annotator
        bar_plot(anno_irrel_times, anno_rel_times, anno_second_times,
                 anno_third_times,
                 "Median annotation time for {} ignoring the first {}"
                 .format(name, k+1), data_type, dst, name=name, color=color)
        # Plot confidence times per annotator
        bar_plot(anno_irrel_conf_times, anno_rel_conf_times,
                 anno_second_conf_times, anno_third_conf_times,
                 "Median confidence time for {} ignoring the first {}"
                 .format(name, k+1), data_type, dst,
                 name=name, is_anno_time=False, color=color)

    # Plot median/average annotation times for the institution
    bar_plot(irrel_times, rel_times, second_times, third_times,
             "Median annotation time ignoring the first {}"
             .format(k+1), data_type, dst, color=color)
    # Plot median/average confidence times for the institution
    bar_plot(irrel_conf_times, rel_conf_times, second_conf_times,
             third_conf_times,
             "Median confidence time ignoring the first {}"
             .format(k+1), data_type, dst, is_anno_time=False, color=color)


def bar_plot(irrel, rel, second, third, title, data_type, dst, name="",
             is_anno_time=True, color="red"):
    """
    Make a bar plot for the given data storing it under the specified path..

    Parameters
    ----------
    irrel: list - values representing "irrelevant" tweets.
    rel: list - values representing "relevant" tweets.
    second: list - values representing 2nd set of labels.
    third: list - values representing 3rd set of labels.
    title: str - title of the plot.
    data_type: str - "raw" or "cleaned".
    dst: str - path under which the plot should be stored.
    name: str - username of the annotator.
    is_anno_time: bool - True if the values represent annotation times.
    Otherwise they are confidence times.
    color: str - color in matplotlib used to fill the bars.

    """
    # Order in which the bars are plotted in the graph
    X_BARS = ["Irrelevant", "Relevant", "Factual/\nNon-factual",
              "Positive/\nNegative"]
    # Compute median/average annotation/confidence times
    avg1, dev1 = _calculate_median_and_mad(irrel)
    avg2, dev2 = _calculate_median_and_mad(rel)
    avg3, dev3 = _calculate_median_and_mad(second)
    avg4, dev4 = _calculate_median_and_mad(third)
    y = [avg1, avg2, avg3, avg4]
    errors = [dev1, dev2, dev3, dev4]
    fig = plt.figure(figsize=(5, 3))
    x = np.arange(len(X_BARS))
    ax = fig.add_subplot(111)
    # plt.title(title)
    # Bar width
    width = 0.3
    # Plot deviations from the mean/median for the entire institution
    if name is None:
        # ax.bar(x, y, width, yerr=errors, color="red")
        ax.bar(x, y, width, color=color)
    else:
        ax.bar(x, y, width, color=color)
    # Set range of axes - use different configs for different datasets
    # to beautify the visualization
    # if len(tweets) <= 50:
    #     plt.xlim(0, len(tweets)+1)
    # else:
    #     plt.xlim(0, len(tweets)+10)
    # plt.ylim(0, max(conf_labels)+10)
    # Hide the right and top spines (lines)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # Set labels of axes
    # ax.set_xlabel("Labels")
    val_type = "confidence"
    if is_anno_time:
        val_type = "annotation"
    ylabel = "Median {} time in s".format(val_type)
    # Add labels for axes and x-axis ticks
    ax.set_ylabel(ylabel)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(X_BARS)
    # Add legend outside of plot
    legend = ax.legend(loc='best', shadow=True,
                       bbox_to_anchor=(0.5, 1.8), fontsize=FONTSIZE)
    username = name
    if name != "":
        fname = "{}_{}_median_times_{}.pdf".format(username, val_type,
                                                   data_type)
    else:
        fname = "{}_times_median_{}.pdf".format(val_type, data_type)
    # Save plot
    plt.savefig(os.path.join(dst, fname), bbox_inches='tight', dpi=600)
    plt.close()


def _calculate_median_and_mad(l):
    """
    Calculates median position of a given group and its median absolute
    deviation (robust version of standard deviation).

    Parameters
    ----------
    l: list of int - each int represents a position at which the tweet was
    labeled by an annotator. Potentially contains np.nan values.

    Returns
    -------
    float, float.
    Median and median absolute deviation of the list.

    """
    med = np.nan
    mad = np.nan
    # Calculate mean iff list isn't empty
    if len(l) > 0:
        vals = np.array(l)
        # Discard np.nan values
        vals = vals[~np.isnan(vals)]
        med = np.median(vals)
        # Use sample deviation (divide by n-1) instead of population
        # deviation (divide by n) since we use only a sample of the whole
        # population
        mad = robust.mad(vals)
    return med, mad


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
    # Number of tweets to ignore in the calculations starting from the beginning
    # to account for the learning effect
    k = 0

    # SU
    # Directory in which figures will be stored
    FIG_DIR = os.path.join(base_dir, "results", "figures",
                           "irrelevant_vs_rest_su_all")
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)

    # SU
    # Use raw and compute median
    plot_irrelevant_vs_rest_per_institution_and_anno(
        su_tweets, su_annos, FIG_DIR, cleaned=False, k=k, color="crimson")
    # Use cleaned and compute median
    plot_irrelevant_vs_rest_per_institution_and_anno(
        su_tweets, su_annos, FIG_DIR, cleaned=True, k=k, color="crimson")

    # MD
    FIG_DIR = os.path.join(base_dir, "results", "figures",
                           "irrelevant_vs_rest_md_all")
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    # MD
    # Use raw and compute median
    plot_irrelevant_vs_rest_per_institution_and_anno(
        md_tweets, md_annos, FIG_DIR, cleaned=False, k=k, color="yellowgreen")
    # Use cleaned and compute median
    plot_irrelevant_vs_rest_per_institution_and_anno(
        md_tweets, md_annos, FIG_DIR, cleaned=True, k=k, color="yellowgreen")
