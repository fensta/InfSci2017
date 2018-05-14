"""
Plotting is done in this script
"""
import os

import matplotlib.pyplot as plt
from scipy import stats
from numpy import array

import pseudo_db


def plot_time_per_tweet_for_anno(tweet_path, anno_path, dst):
    """
    Plots for a single annotator the total times needed to label each tweet.
    Total time is the sum of all recorded times and the plot is stored. The file
    name is of the form: <username_in_db>_<annotator_group>_annotation_times.png

    Parameters
    ----------
    tweet_path: str - path to tweet csv dataset.
    anno_path: str - path to annotator csv dataset.
    dst: str - directory in which the plot will be stored.

    """
    data = pseudo_db.Data(tweet_path, anno_path)
    # For each annotator
    for anno in data.annotators.all_annos():
        # X-axis starting with the tweet that was labeled first (x=1, x=2...x=n)
        # Contains all tweets which is necessary for linear fitting.
        tweets = []
        # Same as <tweets>, but only contains entries for tweets with opinions
        tweets_with_opinion = []
        # Same as <tweets>, but only contains entries for tweets without
        # opinions
        tweets_without_opinion = []
        # Overall annotation times per tweet in s.
        # # Contains all times which is necessary for linear fitting.
        anno_times = []
        # Same as <anno_times>, but only contains annotation times that used 3rd
        # set of labels
        anno_times_with_opinion = []
        # Same as <anno_times>, but only contains annotation times that didn't
        # use 3rd set of labels
        anno_times_without_opinion = []
        group = anno.get_group()
        # For each tweet
        for idx, t in enumerate(anno.get_labeled_tweets()):
            has_opinion = False
            labels = t.get_anno_labels()
            times = t.get_anno_times()
            # Annotation times for hierarchy levels 2 and 3 might be ignored
            t3 = pseudo_db.ZERO
            # First level
            t1 = times[0]
            # Second level
            t2 = times[1]
            l2 = labels[1]
            # Annotator labeled the 3rd set of labels as well
            if l2 == pseudo_db.GROUND_TRUTH["Non-factual"]:
                t3 = times[2]
                has_opinion = True
            # Total time is the maximum time that was needed as it represents
            # the last label/confidence assigned. All counters started as soon
            # as the tweet was displayed.
            total = sum([t1, t2, t3])
            anno_times.append(total)
            tweets.append(idx + 1)
            # Add entry containing labels of 3rd set
            if has_opinion:
                tweets_with_opinion.append(idx + 1)
                anno_times_with_opinion.append(total)
            else:
                # Add entry containing no 3rd label set
                tweets_without_opinion.append(idx + 1)
                anno_times_without_opinion.append(total)
        # Fit a line to the annotator data to see a tendency
        x, y, slope = _lin_fit(tweets, anno_times)
        # Plot annotation time curve for user and store it in
        fig = plt.figure(figsize=(20, 2))
        ax = fig.add_subplot(111)
        # Set the size of the data points
        s = [10 for n in range(len(tweets))]
        # ax.scatter(tweets, anno_times, s=s)
        # Use different colors for annotations containing 3rd set of labels
        # and annotations not containing them
        ax.scatter(tweets_with_opinion, anno_times_with_opinion, color="red",
                   s=s, label="Opinionated annotation")
        ax.scatter(tweets_without_opinion, anno_times_without_opinion,
                   color="green", s=s, label="Neutral annotation")
        # Plot linear regression
        avg_time = sum(anno_times) / len(anno_times)
        ax.plot(x, y, "r-", linewidth=2, color="black",
                label="Avg={:.2f}s; Slope={:.2f}".format(avg_time, slope))
        # Set range of axes - use different configs for different datasets
        # to beautify the visualization
        if len(tweets) == 50:
            plt.xlim(0, len(tweets)+1)
        else:
            plt.xlim(0, len(tweets)+10)
        plt.ylim(0, max(anno_times)+10)
        # Hide the right and top spines (lines)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # Set labels of axes
        ax.set_xlabel("#Annotated tweets")
        ax.set_ylabel("Annotation time in s")
        # Add legend outside of plot
        legend = ax.legend(loc='upper center', shadow=True,
                           bbox_to_anchor=(0.5, 1.8))
        # Just use the part of the username that precedes @
        name = anno.get_name()
        fname = "{}_{}_annotation_times.png".format(name, group)
        # Save plot
        print "#labeled tweets:", len(anno.get_labeled_tweets_ids())
        plt.savefig(os.path.join(dst, fname), bbox_inches='tight', dpi=300)
        plt.close()


# http://glowingpython.blogspot.com.tr/2012/03/linear-regression-with-numpy.html
def _lin_fit(x, y):
    """
    Performs linear regression.

    Parameters
    ----------
    x: list - x-values.
    y: list - y-values.

    Returns
    -------
    list, list, double.
    List of x-values, list of y-values representing the fitted line; slope of
    the fitted line.

    """
    slope, intercept, corr_coeff, p_value, std_err = stats.linregress(array(x),
                                                                      array(y))
    print "r^2: {}, i.e. {}% of the randomness in the data is explained by " \
          "the fitted line.".format(corr_coeff**2, corr_coeff**2 * 100)
    print "p-value:", p_value
    print "Standard error of the estimated gradient:", std_err
    print "Slope:", slope
    line = slope*array(x) + intercept
    return x, line.tolist(), slope

if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))

    # Directory in which figures will be stored
    FIG_DIR = os.path.join(base_dir, "results", "figures",
                           "annotation_curve_su_all")

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
    # SU
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    plot_time_per_tweet_for_anno(su_tweets, su_annos, FIG_DIR)

    # MD
    FIG_DIR = os.path.join(base_dir, "results", "figures",
                           "annotation_curve_md_all")
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    plot_time_per_tweet_for_anno(md_tweets, md_annos, FIG_DIR)
