"""
Examine whether annotators become more confident about the labels they assign
over time. Creates plots.

"""
import os

import matplotlib.pyplot as plt
from scipy import stats
from numpy import array
import numpy as np

import pseudo_db


def plot_confidence_label_per_tweet_for_anno(tweet_path, anno_path, dst,
                                             cleaned=False, k=0):
    """
    Plots for a single annotator the confidence times needed to label each
    tweet.

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
    # Map nominal data to numerical data
    CONF = {
        "Low": 1,
        "High": 2
    }
    data = pseudo_db.Data(tweet_path, anno_path)
    # For each annotator
    for anno in data.annotators.all_annos():
        group = anno.get_group()
        username = anno.get_name()
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
        conf_labels = []
        # Same as <anno_times>, but only contains annotation times that used 3rd
        # set of labels
        conf_labels_with_opinion = []
        # Same as <anno_times>, but only contains annotation times that didn't
        # use 3rd set of labels
        conf_labels_without_opinion = []
        # Annotation times for "irrelevant" tweets
        conf_labels_irrelevant = []
        print "Collect tweets for annotator '{}'".format(username)
        # Number of tweets to skip in the beginning of each annotator
        skipped = 0
        # For each tweet
        for idx,t in enumerate(anno.get_labeled_tweets()):
            labels = t.get_anno_labels()
            c_labels = t.get_conf_labels()
            # After skipping the first k tweets, continue as usual computing
            # confidence times
            if skipped >= k:
                has_opinion = False
                # True, if the tweet isn't "irrelevant"
                has_more = False
                # First level
                rel_label = labels[0]
                c1 = CONF[c_labels[0]]
                # Second and third level might have not been assigned
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
                    c2 = CONF[c_labels[1]]
                    # Annotator labeled the 3rd set of labels as well
                    if l2 == pseudo_db.GROUND_TRUTH["Non-factual"]:
                        # Third level
                        c3 = CONF[c_labels[2]]
                        has_opinion = True
                # Average over all label values
                cs = [c1, c2, c3]
                if has_more and has_opinion:
                    total = 1.0*sum(cs) / 3
                elif has_more and not has_opinion:
                    total = 1.0*sum(cs) / 2
                else:
                    total = sum(cs)
                # print "Irrlevant?", rel_label
                # print "times:", cs
                # print "selected:", total
                tweets.append(idx+1-skipped)
                conf_labels.append(total)
                # Add entry containing labels of 3rd set
                if has_more and has_opinion:
                    tweets_with_opinion.append(idx+1-skipped)
                    conf_labels_with_opinion.append(total)
                elif has_more and not has_opinion:
                    # Add entry containing no 3rd label set
                    tweets_without_opinion.append(idx+1-skipped)
                    conf_labels_without_opinion.append(total)
                else:
                    tweets_irrelevant.append(idx+1-skipped)
                    conf_labels_irrelevant.append(total)
                # print "Labeled:", tweet["text"]
            else:
                # We just skipped another tweet
                skipped += 1
        # Fit a line to the annotator data to see a tendency
        x, y, slope = _lin_fit(tweets, conf_labels)
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
        ax.scatter(tweets_with_opinion, conf_labels_with_opinion,
                   color="darkorange", s=s, label="Opinionated confidence")
        ax.scatter(tweets_without_opinion, conf_labels_without_opinion,
                   color="green", s=s, label="Neutral confidence")
        # Display which tweets were actually "Irrelevant"
        if cleaned:
            ax.scatter(tweets_irrelevant, conf_labels_irrelevant,
                       color="orchid", s=s, label="Irrelevant confidence")
        # Plot linear regression
        avg_time = sum(conf_labels) / len(conf_labels)
        ax.plot(x, y, "r-", linewidth=2, color="black",
                label="Avg={:.2f}s; Slope={:.2f}".format(avg_time, slope))
        # Set range of axes - use different configs for different datasets
        # to beautify the visualization
        if len(tweets) <= 50:
            plt.xlim(0, len(tweets)+1)
        else:
            plt.xlim(0, len(tweets)+10)
        plt.ylim(0, max(conf_labels)+10)
        # Hide the right and top spines (lines)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # Set labels of axes
        ax.set_xlabel("#i-th annotated tweet")
        ax.set_ylabel("Avg. confidence label")
        # Add legend outside of plot
        legend = ax.legend(loc='best', shadow=True,
                           bbox_to_anchor=(0.5, 1.8))
        # Just use the part of the username that precedes @
        name = username.split("@")[0]
        # MD addresses contain "." or rather \u002e, but then the file isn't
        # displayed in PyCharm, so we replace it by "_"
        name = name.replace("\u002e", "_")
        # Find out for which institution this plot is - su or md
        dataset_type = "raw"
        if cleaned:
            dataset_type = "cleaned"
        fname = "{}_{}_confidence_times_{}.png".format(name, group,
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


def plot_avg_conf_label_for_ith_tweet(tweet_path, anno_path, dst,
                                      cleaned=False, k=0):
    """
    Plots how the average confidence label for the i-th tweet in an institution
    averaged over all annotators develop.

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
    dataset_type = "raw"
    if cleaned:
        dataset_type = "cleaned"
    y, annos = get_median_conf_labels(tweet_path, anno_path, cleaned=cleaned,
                                      k=k)
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
    ax.set_ylabel("Average confidence label")
    # Add legend outside of plot
    legend = ax.legend(loc="best", shadow=True, bbox_to_anchor=(0.75, 1.5))
    fname = "{}_ith_tweet_average_confidence_time_{}"\
        .format(institution, dataset_type)
    # Limits of axes
    plt.xlim(-1, x[-1] + 3)
    plt.ylim(0, max(y) + 3)
    plt.savefig(os.path.join(dst, fname), bbox_inches='tight')


def get_median_conf_labels(tweet_path, anno_path, cleaned=False, k=0):
    """
    Collects median confidence labels per i-th tweet in an institution.
    Since annotators labeled different tweets at i-th position, the tweets
    aren't comparable, but the times can be averaged for each i over all
    annotators.

    Parameters
    ----------
    tweet_path: str - path to tweet csv dataset.
    anno_path: str - path to annotator csv dataset.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing median
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
    # Map nominal data to numerical data
    CONF = {
        "Low": 1,
        "High": 2
    }
    # Maximum number of tweets annotated by a single person -> add dummy values
    # (np.nan) if an annotator labeled less tweets
    n = 500 - k
    # List of lists. Each inner list contains all raw annotation times of a
    # single annotator.
    times = []
    data = pseudo_db.Data(tweet_path, anno_path)
    # For each annotator
    for anno in data.annotators.all_annos():
        anno_times = []
        # Number of skipped tweets so far
        skipped = 0
        # For each tweet
        for t in anno.get_labeled_tweets():
            # If enough tweets were skipped, start computing average times
            if k <= skipped:
                labels = t.get_anno_labels()
                c_labels = t.get_conf_labels()
                # Confidence times for hierarchy levels 2 and 3 might have not
                # been assignedbe ignored
                t2 = pseudo_db.ZERO
                t3 = pseudo_db.ZERO
                # First level
                rel_label = labels[0]
                t1 = CONF[c_labels[0]]
                has_opinion = False
                # True, if the tweet isn't "irrelevant"
                has_more = False
                # Discard remaining labels if annotator chose "Irrelevant"
                # Consider other sets of labels only iff either the cleaned
                # dataset should be created and the label is "relevant" OR
                # the raw dataset should be used.
                if (cleaned and rel_label != "Irrelevant") or not cleaned:
                    has_more = True
                    # Second level
                    l2 = labels[1]
                    t2 = CONF[c_labels[1]]
                    # Annotator labeled the 3rd set of labels as well
                    if l2 == pseudo_db.GROUND_TRUTH["Non-factual"]:
                        # Third level
                        t3 = CONF[c_labels[2]]
                        has_opinion = True
                cs = [t1, t2, t3]
                if has_more and has_opinion:
                    total = 1.0 * sum(cs) / 3
                elif has_more and not has_opinion:
                    total = 1.0 * sum(cs) / 2
                else:
                    total = sum(cs)
                anno_times.append(total)
            else:
                # We just skipped another tweet
                skipped += 1

        # Fill up the list with "None" values if less than <n> values are
        # available for an annotator
        anno_times.extend([None] * (n - len(anno_times)))
        times.append(anno_times)
    # m x n matrix that stores n tweets for each of the m annotators. Thus,
    # each row represents the confidence times of a single annotator.
    # If values are missing, np.nan is used.
    times = np.array(times, dtype=float)
    # print times[:, :1]
    # Store n averages; i-th entry corresponds to the confidence time
    # of the i-th tweet averaged over all annotators.
    # Compute average over columns ignoring entries with nan
    median_times = np.nanmean(times, axis=0).T
    return median_times, times.shape[0]


# def collect_data_per_tweet_per_institution(
#         dbs, db_idxs, anno_coll_name="user", tweet_coll_name="tweets",
#         cleaned=False, k=0):
#     """
#     Collects average confidence labels per i-th tweet in an institution.
#     Since annotators labeled different tweets at i-th position, the tweets
#     aren't comparable, but the times can be averaged for each i over all
#     annotators.
#
#     Parameters
#     ----------
#     dbs: list of strings - names of the existing DBs
#     db_idxs: list of ints- name of the MongoDB from where data should be read.
#     anno_coll_name: str - name of the collection holding the annotator data.
#     tweet_coll_name: str - name of the collection holding the tweet data.
#     cleaned: bool - True if the data should be cleaned, i.e. if tweet is
#     "irrelevant", its remaining labels are ignored for computing average
#     annotation times.
#     k: int - number of tweets that should be discarded from the beginning due
#     to the learning effect.
#
#     Returns
#     -------
#     numpy.ndarray, shape(, m); int.
#     Matrix that stores the average confidence times of the first m tweets in
#     the institution. The first entry represents the avg. annotation time of
#     all annotators in the institution for the first tweet. In general, the
#     i-th entry represents the avg. confidence time for the i-th tweet. Since
#     annotators labeled different tweets, a tweet ID can't be derived from that
#     information. If values are missing, np.nan is used.
#     The 2nd returned result is the number of annotators.
#
#     """
#     # Map nominal data to numerical data
#     CONF = {
#         "Low": 1,
#         "High": 2
#     }
#     # Maximum number of tweets annotated by a single person -> add dummy values
#     # (np.nan) if an annotator labeled less tweets
#     n = 500 - k
#     # List of lists. Each inner list contains all raw annotation times of a
#     # single annotator.
#     times = []
#     # Collect data from each DB and aggregate them in the plot at the end
#     for db_idx in db_idxs:
#         # Get DB name
#         db = dbs[db_idx]
#         tweet_coll, anno_coll = utility.load_tweets_annotators_from_db(
#             db, tweet_coll_name, anno_coll_name)
#         DUMMY = 0
#         # For each anno
#         for idx, anno in enumerate(anno_coll.find()):
#             # Overall annotation times for an annotator per tweet in s
#             anno_times = []
#             username = anno["username"]
#             print "Collect tweets for annotator '{}'".format(username)
#             # Tweet IDs labeled by this annotator
#             labeled = anno["annotated_tweets"]
#             # Number of skipped tweets so far
#             skipped = 0
#             for tid in labeled:
#                 has_opinion = False
#                 # True, if the tweet isn't "irrelevant"
#                 has_more = False
#                 # If enough tweets were skipped, start computing average times
#                 if k <= skipped:
#                     tweet = utility.get_tweet(tweet_coll, tid)
#                     rel_label = tweet["relevance_label"][username]
#                     c1 = CONF[tweet["confidence_relevance"][username]]
#                     # Discard remaining labels if annotator chose "Irrelevant"
#                     # Consider other sets of labels only iff either the cleaned
#                     # dataset should be created and the label is "relevant" OR
#                     # the raw dataset should be used.
#                     if (cleaned and rel_label != "Irrelevant") or not cleaned:
#                         has_more = True
#                         c2 = CONF[tweet["confidence_fact"][username]]
#                         # Annotator labeled the 3rd set of labels as well
#                         if username in tweet["opinion_label"]:
#                             c3 = CONF[tweet["confidence_opinion"][username]]
#                             has_opinion = True
#                         else:
#                             # 3rd set of labels might not have been assigned by
#                             # annotator, so choose some low constants that max()
#                             # calculations won't get affected
#                             c3 = DUMMY
#                     else:
#                         # Ignore remaining labels
#                         c2 = DUMMY
#                         c3 = DUMMY
#                     cs = [c1, c2, c3]
#                     if has_more and has_opinion:
#                         total = 1.0*sum(cs) / 3
#                     elif has_more and not has_opinion:
#                         total = 1.0*sum(cs) / 2
#                     else:
#                         total = sum(cs)
#                     anno_times.append(total)
#                 else:
#                     # We just skipped another tweet
#                     skipped += 1
#             # Fill up the list with "None" values if less than <n> values are
#             # available for an annotator
#             anno_times.extend([None] * (n - len(anno_times)))
#             times.append(anno_times)
#
#     # m x n matrix that stores n tweets for each of the m annotators. Thus,
#     # each row represents the confidence times of a single annotator.
#     # If values are missing, np.nan is used.
#     times = np.array(times, dtype=float)
#     # print times[:, :1]
#     # Store n averages; i-th entry corresponds to the confidence time
#     # of the i-th tweet averaged over all annotators.
#     # Compute average over columns ignoring entries with nan
#     avg_times = np.nanmean(times, axis=0).T
#     print avg_times.shape
#     return avg_times, times.shape[0]


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
                           "confidence_label_curve_su_all")
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)

    plot_confidence_label_per_tweet_for_anno(su_tweets, su_annos, FIG_DIR, k=k,
                                             cleaned=False)
    plot_confidence_label_per_tweet_for_anno(su_tweets, su_annos, FIG_DIR, k=k,
                                             cleaned=True)
    # Raw
    plot_avg_conf_label_for_ith_tweet(su_tweets, su_annos, FIG_DIR, k=k,
                                      cleaned=False)
    # Cleaned
    plot_avg_conf_label_for_ith_tweet(su_tweets, su_annos, FIG_DIR, k=k,
                                      cleaned=True)

    # MD
    FIG_DIR = os.path.join(base_dir, "results", "figures",
                           "confidence_label_curve_md_all")
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    plot_confidence_label_per_tweet_for_anno(md_tweets, md_annos, FIG_DIR, k=k,
                                             cleaned=False)
    plot_confidence_label_per_tweet_for_anno(md_tweets, md_annos, FIG_DIR, k=k,
                                             cleaned=True)
    # Raw
    plot_avg_conf_label_for_ith_tweet(md_tweets, md_annos, FIG_DIR, k=k,
                                      cleaned=False)
    # Cleaned
    plot_avg_conf_label_for_ith_tweet(md_tweets, md_annos, FIG_DIR, k=k,
                                      cleaned=True)
