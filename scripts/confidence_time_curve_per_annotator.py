"""
Plots for each annotator a curve displaying their total time needed to rate
their confidences per tweet.
"""
import os

import matplotlib.pyplot as plt
from scipy import stats
from numpy import array
import numpy as np

import pseudo_db


def plot_confidence_per_tweet_for_anno(
        tweet_path, anno_path, dst, cleaned=False, k=0):
    """
    Plots for a single annotator the median confidence times needed to label
    each tweet.

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

    """
    # k actually represents the index and we start counting at 0, hence -1
    k -= 1
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
        # "Irrelevant" tweets
        tweets_irrelevant = []
        # Overall annotation times per tweet in s.
        # # Contains all times which is necessary for linear fitting.
        anno_times = []
        # Same as <anno_times>, but only contains annotation times that used 3rd
        # set of labels
        anno_times_with_opinion = []
        # Same as <anno_times>, but only contains annotation times that didn't
        # use 3rd set of labels
        anno_times_without_opinion = []
        # Annotation times for "irrelevant" tweets
        anno_times_irrelevant = []
        username = anno.get_name()
        group = anno.get_group()
        print "Collect tweets for annotator '{}'".format(username)
        # Number of tweets to skip in the beginning of each annotator
        skipped = 0
        # For each tweet
        for idx, t in enumerate(anno.get_labeled_tweets()):
            # After skipping the first k tweets, continue as usual computing
            # confidence times
            if skipped >= k:
                has_opinion = False
                # True, if the tweet isn't "irrelevant"
                has_more = False
                labels = t.get_anno_labels()
                times = t.get_conf_times()
                # First level
                rel_label = labels[0]
                c1 = times[0]
                # Confidence times for 2nd and 3rd level might be empty
                c2 = pseudo_db.ZERO
                c3 = pseudo_db.ZERO

                # Discard remaining labels if annotator chose "Irrelevant"
                # Consider other sets of labels only iff either the cleaned
                # dataset should be created and the label is "relevant" OR
                # the raw dataset should be used.
                if (cleaned and rel_label != "Irrelevant") or not cleaned:
                    has_more = True
                    # Second level
                    l2 = labels[1]
                    c2 = times[1]
                    # Annotator labeled the 3rd set of labels as well
                    if l2 == pseudo_db.GROUND_TRUTH["Non-factual"]:
                        # Third level
                        c3 = times[2]
                        has_opinion = True
                # Sum up relative times
                cs = [c1, c2, c3]
                total = sum(cs)
                # print "Irrlevant?", rel_label
                # print "times:", cs
                # print "selected:", total
                tweets.append(idx+1-skipped)
                anno_times.append(total)
                # Add entry containing labels of 3rd set
                if has_more and has_opinion:
                    tweets_with_opinion.append(idx+1-skipped)
                    anno_times_with_opinion.append(total)
                elif has_more and not has_opinion:
                    # Add entry containing no 3rd label set
                    tweets_without_opinion.append(idx+1-skipped)
                    anno_times_without_opinion.append(total)
                else:
                    tweets_irrelevant.append(idx+1-skipped)
                    anno_times_irrelevant.append(total)
                # print "Labeled:", tweet["text"]
            else:
                # We just skipped another tweet
                skipped += 1
        # Fit a line to the annotator data to see a tendency
        x, y, slope = _lin_fit(tweets, anno_times)
        # Plot annotation time curve for user and store it in
        fig = plt.figure(figsize=(20, 3))
        ax = fig.add_subplot(111)
        # Set the size of the data points
        s = [10 for n in range(len(tweets))]
        # Title
        title = "Confidence assignment times ignoring the first {} tweets"\
            .format(k+1)
        plt.title(title)
        # ax.scatter(tweets, anno_times, s=s)
        # Use different colors for annotations containing 3rd set of labels
        # and annotations not containing them
        ax.scatter(tweets_with_opinion, anno_times_with_opinion,
                   color="darkorange", s=s, label="Opinionated confidence")
        ax.scatter(tweets_without_opinion, anno_times_without_opinion,
                   color="green", s=s, label="Neutral confidence")
        # Display which tweets were actually "Irrelevant"
        if cleaned:
            ax.scatter(tweets_irrelevant, anno_times_irrelevant,
                       color="orchid", s=s, label="Irrelevant confidence")
        # Plot linear regression
        avg_time = sum(anno_times) / len(anno_times)
        ax.plot(x, y, "r-", linewidth=2, color="black",
                label="Avg={:.2f}s; Slope={:.2f}".format(avg_time, slope))
        # Set range of axes - use different configs for different datasets
        # to beautify the visualization
        if len(tweets) <= 50:
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
        ax.set_ylabel("Confidence assignment time in s")
        # Add legend outside of plot
        legend = ax.legend(loc='best', shadow=True,
                           bbox_to_anchor=(0.5, 1.8))
        # Find out for which institution this plot is - su or md
        dataset_type = "raw"
        if cleaned:
            dataset_type = "cleaned"
        fname = "{}_{}_confidence_times_{}.png".format(username, group,
                                                       dataset_type)
        # Save plot
        print "#labeled tweets:", len(anno.get_labeled_tweets())
        plt.savefig(os.path.join(dst, fname), bbox_inches='tight', dpi=300)


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


def plot_avg_conf_time_for_ith_tweet(tweet_path, anno_path, dst, cleaned=False,
                                     k=0):
    """
    Plots how the average confidence times for the i-th tweet in an institution
    averaged over all annotators.

    Parameters
    ----------
    tweet_path: str - path to tweet csv dataset.
    anno_path: str - path to annotator csv dataset.
    dst: str - directory in which the plot will be stored.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing average
    annotation times.
    k: int - number of tweets that should be discarded from the beginning due
    to the learning effect.

    """
    # k actually represents the index and we start counting at 0, hence -1
    k -= 1
    # Find out for which institution this plot is - su or md
    data = pseudo_db.Data(tweet_path, anno_path)
    institution = data.annotators.get_ith_annotator(0).get_dataset_name()
    print "institution", institution
    dataset_type = "raw"
    if cleaned:
        dataset_type = "cleaned"
    y, annos = collect_data_per_tweet_per_institution(tweet_path, anno_path,
                                                      cleaned=cleaned, k=k)
    # Plotting
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    s = [10 for n in range(annos)]
    x = range(len(y))
    ax.scatter(x, y, color="red", s=s, label="Avg. annotation time")
    # Fit a line to the times, f(x) = mx + n
    xf, yf, m = _lin_fit(x, y)
    ax.plot(xf, yf, color="black", linewidth=2, label="slope={:4.2f}".format(m))
    # Plot the fit
    # Hide the right and top spines (lines)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Title - k+1 because we start with k=0
    title = "Avg. confidence time for {} ignoring the first {} tweets"\
        .format(institution.upper(), k+1)
    plt.title(title)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # Set labels of axes
    ax.set_xlabel("i-th annotated tweet")
    ax.set_ylabel("Average confidence time in s")
    # Add legend outside of plot
    legend = ax.legend(loc="best", shadow=True, bbox_to_anchor=(0.75, 1.5))
    fname = "{}_ith_tweet_average_confidence_time_{}"\
        .format(institution, dataset_type)
    # Limits of axes
    plt.xlim(-1, x[-1] + 3)
    plt.ylim(0, max(y) + 3)
    plt.savefig(os.path.join(dst, fname), bbox_inches='tight')


def collect_data_per_tweet_per_institution(tweet_path, anno_path,
                                           cleaned=False, k=0):
    """
    Collects average confidence times per i-th tweet in an institution.
    Since annotators labeled different tweets at i-th position, the tweets
    aren't comparable, but the times can be averaged for each i over all
    annotators.

    Parameters
    ----------
    tweet_path: str - path to tweet csv dataset.
    anno_path: str - path to annotator csv dataset.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing average
    annotation times.
    k: int - number of tweets that should be discarded from the beginning due
    to the learning effect.

    Returns
    -------
    numpy.ndarray, shape(, m); int.
    Matrix that stores the average confidence times of the first m tweets in
    the institution. The first entry represents the avg. annotation time of
    all annotators in the institution for the first tweet. In general, the
    i-th entry represents the avg. confidence time for the i-th tweet. Since
    annotators labeled different tweets, a tweet ID can't be derived from that
    information. If values are missing, np.nan is used.
    The 2nd returned result is the number of annotators.

    """
    # Maximum number of tweets annotated by a single person -> add dummy values
    # (np.nan) if an annotator labeled less tweets
    n = 500 - k
    # List of lists. Each inner list contains all raw annotation times of a
    # single annotator.
    times = []
    data = pseudo_db.Data(tweet_path, anno_path)
    # Collect data from each DB and aggregate them in the plot at the end
    # For each annotator
    for anno in data.annotators.all_annos():
        # Overall annotation times for an annotator per tweet in s
        anno_times = []
        username = anno.get_name()
        print "Collect tweets for annotator '{}'".format(username)
        # Number of skipped tweets so far
        skipped = 0
        # For each tweet
        for idx, t in enumerate(anno.get_labeled_tweets()):
            has_opinion = False
            print "tid", t.get_tid()
            # True, if the tweet isn't "irrelevant"
            has_more = False
            # If enough tweets were skipped, start computing median times
            if k < skipped:
                labels = t.get_anno_labels()
                conf_times = t.get_conf_times()
                # First level
                rel_label = labels[0]
                c1 = conf_times[0]
                # 2nd and 3rd level might have no labels
                c2 = pseudo_db.ZERO
                c3 = pseudo_db.ZERO
                # Discard remaining labels if annotator chose "Irrelevant"
                # Consider other sets of labels only iff either the cleaned
                # dataset should be created and the label is "relevant" OR
                # the raw dataset should be used.
                if (cleaned and rel_label != "Irrelevant") or not cleaned:
                    has_more = True
                    # Second level
                    l2 = labels[1]
                    c2 = conf_times[1]
                    # Annotator labeled the 3rd set of labels as well
                    if l2 == pseudo_db.GROUND_TRUTH["Non-factual"]:
                        c3 = conf_times[2]
                        has_opinion = True
                cs = [c1, c2, c3]
                # Sum up relative times
                if has_more and has_opinion:
                    total = 1.0*sum(cs) / 3
                elif has_more and not has_opinion:
                    total = 1.0*sum(cs) / 2
                else:
                    total = sum(cs)
                anno_times.append(total)
            else:
                # We just skipped another tweet
                skipped += 1
        # Fill up the list with "None" values if less than <n> values are
        # available for an annotator
        to_add = n - len(anno_times)
        anno_times.extend([None] * to_add)
        times.append(anno_times)
        # Make sure that each list contains exactly <n> entries
        assert(len(anno_times) == n)

    # m x n matrix that stores n tweets for each of the m annotators. Thus,
    # each row represents the confidence times of a single annotator.
    # If values are missing, np.nan is used.
    times = np.array(times, dtype=float)
    # print times[:, :1]
    # Store n averages; i-th entry corresponds to the confidence time
    # of the i-th tweet averaged over all annotators.
    # Compute average over columns ignoring entries with nan
    avg_times = np.nanmean(times, axis=0).T
    print avg_times.shape
    return avg_times, times.shape[0]


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
    k = 20

    # SU
    # Directory in which figures will be stored
    FIG_DIR = os.path.join(base_dir, "results", "figures",
                           "confidence_curve_su_all")
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    # Raw
    plot_confidence_per_tweet_for_anno(su_tweets, su_annos, FIG_DIR,
                                       cleaned=False, k=k)
    # Cleaned
    plot_confidence_per_tweet_for_anno(su_tweets, su_annos, FIG_DIR,
                                       cleaned=True, k=k)
    # Raw
    plot_avg_conf_time_for_ith_tweet(su_tweets, su_annos, FIG_DIR,
                                     cleaned=False, k=k)
    # Cleaned
    plot_avg_conf_time_for_ith_tweet(su_tweets, su_annos, FIG_DIR,
                                     cleaned=True, k=k)

    # MD
    FIG_DIR = os.path.join(base_dir, "results", "figures",
                           "confidence_curve_md_all")
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    # Raw
    plot_confidence_per_tweet_for_anno(md_tweets, md_annos, FIG_DIR,
                                       cleaned=False, k=k)
    # Cleaned
    plot_confidence_per_tweet_for_anno(md_tweets, md_annos, FIG_DIR,
                                       cleaned=True, k=k)
    # Raw
    plot_avg_conf_time_for_ith_tweet(md_tweets, md_annos, FIG_DIR,
                                     cleaned=False, k=k)
    # Cleaned
    plot_avg_conf_time_for_ith_tweet(md_tweets, md_annos, FIG_DIR,
                                     cleaned=True, k=k)
