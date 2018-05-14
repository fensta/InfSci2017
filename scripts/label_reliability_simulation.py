"""
Run simulation to test if reliability of labels increases after the learning
phase:
- use all k tweets within the learning phase to to learn a model and then
predict the remaining tweets
- use the k last tweets from an annotator to learn a model and then predict
the remaining tweets - we additionally ignore all tweets from the learning
phase
- we expect the performance to be higher after ignoring the first k tweets and
learning from the last k tweets as this would suggest that the labels become
more reliable toward the end

"""
import os

import matplotlib.pyplot as plt

import pseudo_db
from knn import knn, hierarchical_precision_recall_f1, compute_micro_metrics

# Value representing a missing label
# NA = " "

# For Information sciences
FONTSIZE = 12
plt.rcParams.update({'font.size': FONTSIZE})


def run_simulation(intervals, k_max, k_s, k_m, stat_dst, md_tw_p, md_a_p,
                   su_tw_p, su_a_p, cleaned=False, weights="uniform",
                   recency=False, keep_na=False):
    """
    Tests if label reliability increases after learning phase.

    Parameters
    ----------
    intervals: list of tuples - each list represents an interval. First entry
    is for group S, 2nd  entry for M with
    fatigue (without fatigue can be derived from it by simply adding everything
    after learning phase to rest). Each interval contains the boundaries.
    k_max: int - maximum number of neighbors to consider when predicting label.
    It starts always with 1.
    k_s: Number of tweets to see before learning phase in S is completed.
    k_m: Number of tweets to see before learning phase in M is completed.
    stat_dst: str - directory in which the statistics will be stored.
    md_tw_p: str - path to MD tweets dataset in csv format.
    md_a_p: str - path to MD annotators dataset in csv format.
    su_tw_p: str - path to SU tweets dataset in csv format.
    su_a_p: str - path to SU annotators dataset in csv format.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing average
    annotation times.
    weights: str - weight function used in prediction.  Possible values:
        - "uniform" : uniform weights.  All points in each neighborhood
          are weighted equally.
        - "distance" : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
    recency: bool - True if scenario 2 should be tested. Otherwise scenario 1
    is tested.
    keep_na: bool - True if neighbors with no label should be considered when
    predicting the label of an unknown instance. If False, such instances with
    unavailable labels are discarded.

    """
    agg = "raw"
    if cleaned:
        agg = "cleaned"
    s = intervals[0]
    m_fatigue = intervals[1]
    # Below is an example for the datastructure holding the data used for kNN
    # {
    #     "md":
    #         {
    #             #
    #             "S": {
    #                 "anno1":
    #                     [
    #                         # Learning phase contains all tweets that anno
    #                         # labeled within learning phase = Interval 1
    #                         [
    #                             # tweets labeled
    #                             [
    #                                 # all labels for tweet 1, plus tweet text
    #                                 # at end
    #                                 [l1, l2, l3, "tweet1 text"],
    #                                 [l1, "tweet2 text"],...
    #                             ]
    #                         ],
    #                         # All tweets labeled after learning phase by anno
    #                         # = interval 2
    #                         [
    #                             # tweets labeled
    #                             [
    #                                 # all labels for tweet 1, plus tweet text
    #                                 # at end
    #                                 [l1, l2, l3, "tweet1 text"],
    #                                 [l1, "tweet2 text"],...
    #                             ]
    #                         ]
    #                     ],
    #                 "anno2": [...]
    #             },
    #             "M": {
    #                 "anno1":
    #                     [
    #                         # Learning phase contains all tweets that anno
    #                         # labeled within learning phase = Interval 1
    #                         [
    #                             # tweets labeled
    #                             [
    #                                 # all labels for tweet 1, plus tweet text
    #                                 # at end
    #                                 [l1, l2, l3, "tweet1 text"],
    #                                 [l1, "tweet2 text"],...
    #                             ]
    #                         ],
    #                         # All tweets labeled after learning phase by anno
    #                         # = M without fatigue = M with fatigue (same in
    #                         # both cases)
    #                         [
    #                             # tweets labeled
    #                             [
    #                                 # all labels for tweet 1, plus tweet text
    #                                 # at end
    #                                 [l1, l2, l3, "tweet1 text"],
    #                                 [l1, "tweet2 text"],...
    #                             ]
    #                         ],
    #                         # All tweets labeled after learning phase by anno
    #                         # = M with fatigue = 2nd interval
    #                         [
    #                             # tweets labeled
    #                             [
    #                                 # all labels for tweet 1, plus tweet text
    #                                 # at end
    #                                 [l1, l2, l3, "tweet1 text"],
    #                                 [l1, "tweet2 text"],...
    #                             ]
    #                         ],
    #                         # All tweets labeled after learning phase by anno
    #                         # = M with fatigue rest = 3rd interval
    #                         [
    #                             # tweets labeled
    #                             [
    #                                 # all labels for tweet 1, plus tweet text
    #                                 # at end
    #                                 [l1, l2, l3, "tweet1 text"],
    #                                 [l1, "tweet2 text"],...
    #                             ]
    #                         ],
    #                     ],
    #                 "anno2": [...]
    #             }
    #         },
    #     "su": see "md"
    # }
    labels = {
        "md": collect_data_per_annotator_per_group(md_tw_p, md_a_p, s,
                                                   m_fatigue, cleaned=cleaned),
        "su": collect_data_per_annotator_per_group(su_tw_p, su_a_p, s,
                                                   m_fatigue, cleaned=cleaned)
    }
    data = {}
    # learning_first_naive_try(labels, weights, recency, k, k_s, k_m)
    # For each similarity measure
    for similarity in ["substring", "subsequence", "edit"]:
    # The next line is only used when searching for good examples of nearest
    # neighbors for the paper -> faster than using all distance functions
    # for similarity in ["edit"]:
        data[similarity] = {}
        # Only use odd k values to avoid ties in knn()
        for k in xrange(1, k_max+1, 2):
            result = hierarchical_learning(labels, weights, recency, k, k_s,
                                           k_m, keep_na, similarity)
            data[similarity][k] = result

    # Store infsci2017_results in a csv file
    fname = "label_reliability_k_{}_k_s_{}_k_m_{}_{}.txt"\
        .format(k_max, k_s, k_m, agg)

    store_csv(data, stat_dst, fname)


def store_csv(data, stat_dst, fname):
    """
    Stores data in 2 csv files, one per institution.

    Parameters
    ----------
    data: dict - see end of hierarchical_learning for proper format of the
    nested dict.
    stat_dst: str - directory in which the resulting csv file will be stored.
    fname: str - name of the csv file, which will be slightly changed inside
    this function.

    """
    for inst in ["md", "su"]:
        lines = []
        dst = os.path.join(stat_dst, "{}_".format(inst) + fname)
        for similarity in data:
            for k in data[similarity]:
                m1 = data[similarity][k][inst]["learn"]["S"]
                m2 = data[similarity][k][inst]["learn"]["M"]
                m3 = data[similarity][k][inst]["learn"]["All"]
                m4 = data[similarity][k][inst]["rest"]["S"]
                m5 = data[similarity][k][inst]["rest"]["M"]
                m6 = data[similarity][k][inst]["rest"]["All"]
                # k, interval, similarity, group, precision, recall, f1
                lines.append([k, "learn", similarity, "S", m1[0], m1[1], m1[2]])
                lines.append([k, "rest", similarity, "S", m4[0], m4[1], m4[2]])
                lines.append([k, "learn", similarity, "M", m2[0], m2[1], m2[2]])
                lines.append([k, "rest", similarity, "M", m5[0], m5[1], m5[2]])
                lines.append([k, "learn", similarity, "All", m3[0], m3[1],
                             m3[2]])
                lines.append([k, "rest", similarity, "All", m6[0], m6[1],
                             m6[2]])

        with open(dst, "wb") as f:
            # Header
            f.write("neighbors, interval, similarity function, group, "
                    "precision, recall, f1\n")
            for line in lines:
                f.write("{},{},{},{},{},{},{}\n"
                        .format(line[0], line[1], line[2], line[3], line[4],
                                line[5], line[6]))


def read_data_from_csv(src, metric_name, group, similarity):
    """
    Read in a csv file that stored the infsci2017_results of one of the
    experiments.

    Parameters
    ----------
    src: str - location of input csv file.
    metric_name: str - name of the hierarchical metric: "prec" precision,
    "rec" for recall, or "f1" for F1-score
    group: str - annotator group that should be plotted: "All" for
    institution, "S" for group S, "M" for group M
    similarity: str - name of the metric to be displayed: "subsequence" for
    longest common subsequence, "substring" for longest common substring,
    "edit" for edit distance.

    Returns
    -------
    List of int, list of float, list of int, list of float.
    Returns the filtered data as lists of x and y values.  On x-axis, k is
    displayed and on y the respective metric. Returns x,y values for learning
    phase and rest separately, hence 4 return values instead of two.

    """
    with open(src, "rb") as f:
        lines = f.readlines()
    x_learn = []
    y_learn = []
    x_rest = []
    y_rest = []
    # k's aren't sorted in csv file, so we must do this
    next_k = 1
    next_interval = "learn"
    next_line_was_found = True
    # Find next line from which data to be plotted should be extracted
    # Stops when desired line doesn't exist
    while next_line_was_found:
        next_line_was_found = False
        # Skip header
        for line in lines[1:]:
            content = line.split(",")
            # Each line stores the following info:
            # k, interval, similarity, group, precision, recall, f1
            k = int(content[0])
            interval = content[1]
            func = content[2]
            gr = content[3]
            # Line was found
            if k == next_k and interval == next_interval and gr == group \
                    and func == similarity:
                # Get performance metric value
                if metric_name == "prec":
                    y = float(content[4])
                elif metric_name == "rec":
                    y = float(content[5])
                else:
                    y = float(content[6])
                if interval == "learn":
                    x_learn.append(k)
                    y_learn.append(y)
                    # Update variable for next line
                    next_interval = "rest"
                else:
                    x_rest.append(k)
                    y_rest.append(y)
                    # Update variable for next line
                    next_interval = "learn"
                next_line_was_found = True
                # Update variables: if we just found "learn", we need "rest"
                # interval next
                if next_interval == "learn":
                    # We use only odd intervals
                    next_k += 2
    return x_learn, y_learn, x_rest, y_rest


def hierarchical_learning(labels, weights, recency, k, k_s, k_m, keep_na,
                          similarity):
    """
    Cast the learning problem as a hierarchical classification task.
    For each hierarchy level, a separate classifier is trained. It's impossible
    to build a classifier for each parent class due to a potential lack of
    training examples (e.g. for 2nd level only "relevant" tweets exist, so no
    classifier could be built for 2nd level of "irrelevant"). Therefore, in
    total 6 classifiers are created per user: 2 per level and there are 3
    levels. 2 per level because we need one classifier for learning phase and
    one for rest. Each classifier is invoked and for evaluation micro-averaged
    hierarchical precision, recall and F1-score are leveraged.


    Parameters
    ----------
    labels: dict of dict of dict of list of list of tuple - Tuple contains 3
    labels for a tweet assigned by an annotator, plus the actual tweet. Next
    list represents all labeled tweets by annotator. Next list represents
    intervals, learning phase and rest. Next dict contains annotator names, the
    next one the annotator groups and the top-level dict contains the
    institutions.
    weights: str - weight function used in prediction.  Possible values:
        - "uniform" : uniform weights.  All points in each neighborhood
          are weighted equally.
        - "distance" : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
    recency: bool - True if scenario 2 should be tested. Otherwise scenario 1
    is tested.
    k: int - number of neighbors to consider when predicting label.
    k_s: Number of tweets to see before learning phase in S is completed.
    k_m: Number of tweets to see before learning phase in M is completed.
    keep_na: bool - True if neighbors with no label should be considered when
    predicting the label of an unknown instance. If False, such instances with
    unavailable labels are discarded.
    similarity: str - similarity function to be used:
        - "substring": longest common SUBSTRING (= contiguously shared words)
        - "subsequence": longest common SUBSEQUENCE (= shared words in relative
        order, not necessarily contiguous)
        - "edit": edit distance

    Returns
    -------
    dict of dicts of dicts.
    Dictionary holds the micro-averaged performance metrics per group and
    institution. Keys are "md" and "su" and in each inner dict we have for
    learning phase "learning" and rest "rest" another dict as follows.
    Keys  are the groups, "S","M" and "All" and their values are hierarchical
    precision, recall, and F1-score in that order in a list. "All" represents
    an institution's performance metrics.

    """
    # Store performance metrics that should be returned at the end
    all_stats = {
        "md": {
            "learn": {
                    "S": [],
                    "M": [],
                    "All": []
            },
            "rest": {
                    "S": [],
                    "M": [],
                    "All": []
            }
        },
        "su": {
            "learn": {
                    "S": [],
                    "M": [],
                    "All": []
            },
            "rest": {
                    "S": [],
                    "M": [],
                    "All": []
            }
        }
    }
    # For each institution
    for inst in labels:
        print "institution", inst
        # To compute micro-averaged statistics per institution, store exact
        # counts for tweets of learning phase and remaining tweets separately
        micro_stats_inst_learn = []
        micro_stats_inst_rest = []
        # For each annotator group
        for i, group in enumerate(labels[inst]):
            print "group:", group
            # To compute micro-averaged statistics per group, store exact counts
            # Store tweets within learning phase separately from remaining ones
            micro_stats_group_learn = []
            micro_stats_group_rest = []
            # For each annotator
            for anno in labels[inst][group]:
                # print "anno:", anno, "#intervals:", \
                #     len(labels[inst][group][anno])
                intervls = labels[inst][group][anno]
                if group == "S":
                    s_learn = intervls[0]
                    s_rest = intervls[1]
                    # print "s_learn", len(s_learn)
                    # print "s_rest", len(s_rest)
                    # Train on learning phase and test on rest
                    ##########################################
                    # Extract training and test data
                    y_train_first, y_train_second, y_train_third, train = \
                        extract_labels_text(s_learn)
                    y_true_first, y_true_second, y_true_third, test = \
                        extract_labels_text(s_rest)
                    # print "train", len(train)
                    # print "test",len(test)

                    # Train/test once per label set
                    # First label set
                    y_preds1 = knn(train, y_train_first, test, k=k,
                                   recency=recency, weights=weights,
                                   keep_na=keep_na, similarity=similarity)
                    # print "preds 1", len(y_preds1)
                    # print y_preds1

                    # Second label set
                    y_preds2 = knn(train, y_train_second, test, k=k,
                                   recency=recency, weights=weights,
                                   keep_na=keep_na, similarity=similarity)
                    # print "preds2", len(y_preds2)
                    # print y_preds2
                    # Third label set
                    y_preds3 = knn(train, y_train_third, test, k=k,
                                   recency=recency, weights=weights,
                                   keep_na=keep_na, similarity=similarity)
                    # print "preds3", len(y_preds3)
                    # print y_preds3
                    # So far labels are predicted per level, but for evaluation
                    # it's expected to group them per annotator (i.e. all levels
                    # per annotator)
                    y_preds = [y_preds1, y_preds2, y_preds3]
                    y_true = [y_true_first, y_true_second, y_true_third]
                    y_preds, y_true = aggregate_labels_per_anno(y_preds, y_true)

                    # print "aggregated y_preds", len(y_preds)
                    # print y_preds

                    _, _, _, stat = hierarchical_precision_recall_f1(
                        y_preds, y_true)
                    micro_stats_group_learn.append(stat)
                    micro_stats_inst_learn.append(stat)
                    # print stat

                    # Train on last k tweets and test on rest
                    # (without learning phase)
                    ##########################################
                    # Extract training and test data
                    y_train_first, y_train_second, y_train_third, train = \
                        extract_labels_text(s_rest[-k_s:])
                    y_true_first, y_true_second, y_true_third, test = \
                        extract_labels_text(s_rest[:-k_s])

                    # Train/test once per label set
                    # First label set
                    y_preds1 = knn(train, y_train_first, test, k=k,
                                   recency=recency, weights=weights,
                                   keep_na=keep_na, similarity=similarity)

                    # Second label set
                    y_preds2 = knn(train, y_train_second, test, k=k,
                                   recency=recency, weights=weights,
                                   keep_na=keep_na, similarity=similarity)

                    # Third label set
                    y_preds3 = knn(train, y_train_third, test, k=k,
                                   recency=recency, weights=weights,
                                   keep_na=keep_na, similarity=similarity)

                    # So far labels are predicted per level, but for evaluation
                    # it's expected to group them per annotator (i.e. all levels
                    # per annotator)
                    y_preds = [y_preds1, y_preds2, y_preds3]
                    y_true = [y_true_first, y_true_second, y_true_third]
                    y_preds, y_true = aggregate_labels_per_anno(y_preds, y_true)
                    _, _, _, stat = hierarchical_precision_recall_f1(
                        y_preds, y_true)
                    # print stat
                    micro_stats_group_rest.append(stat)
                    micro_stats_inst_rest.append(stat)
                else:
                    m_learn = intervls[0]
                    m_rest = intervls[1]
                    # print "m_learn", len(m_learn)
                    m_fat = intervls[2]
                    m_fat_rest = intervls[3]
                    # Train on learning phase and test on rest
                    ##########################################
                    # Extract training and test data
                    y_train_first, y_train_second, y_train_third, train = \
                        extract_labels_text(m_learn)
                    y_true_first, y_true_second, y_true_third, test = \
                        extract_labels_text(m_rest)

                    # Train/test once per label set
                    # First label set
                    y_preds1 = knn(train, y_train_first, test, k=k,
                                   recency=recency, weights=weights,
                                   keep_na=keep_na, similarity=similarity)

                    # Second label set
                    y_preds2 = knn(train, y_train_second, test, k=k,
                                   recency=recency, weights=weights,
                                   keep_na=keep_na, similarity=similarity)

                    # Third label set
                    y_preds3 = knn(train, y_train_third, test, k=k,
                                   recency=recency, weights=weights,
                                   keep_na=keep_na, similarity=similarity)

                    # So far labels are predicted per level, but for evaluation
                    # it's expected to group them per annotator (i.e. all levels
                    # per annotator)
                    y_preds = [y_preds1, y_preds2, y_preds3]
                    y_true = [y_true_first, y_true_second, y_true_third]
                    y_preds, y_true = aggregate_labels_per_anno(y_preds, y_true)
                    _, _, _, stat = hierarchical_precision_recall_f1(
                        y_preds, y_true)
                    # print stat
                    micro_stats_group_learn.append(stat)
                    micro_stats_inst_learn.append(stat)

                    # Train on last k tweets and test on rest
                    # (without learning phase)
                    ##########################################
                    # Extract training and test data
                    y_train_first, y_train_second, y_train_third, train = \
                        extract_labels_text(m_rest[-k_m:])
                    y_true_first, y_true_second, y_true_third, test = \
                        extract_labels_text(m_rest[:-k_m])

                    # Train/test once per label set
                    # First label set
                    y_preds1 = knn(train, y_train_first, test, k=k,
                                   recency=recency, weights=weights,
                                   keep_na=keep_na, similarity=similarity)

                    # Second label set
                    y_preds2 = knn(train, y_train_second, test, k=k,
                                   recency=recency, weights=weights,
                                   keep_na=keep_na, similarity=similarity)

                    # Third label set
                    y_preds3 = knn(train, y_train_third, test, k=k,
                                   recency=recency, weights=weights,
                                   keep_na=keep_na, similarity=similarity)

                    # So far labels are predicted per level, but for evaluation
                    # it's expected to group them per annotator (i.e. all levels
                    # per annotator)
                    y_preds = [y_preds1, y_preds2, y_preds3]
                    y_true = [y_true_first, y_true_second, y_true_third]
                    y_preds, y_true = aggregate_labels_per_anno(y_preds, y_true)
                    _, _, _, stat = hierarchical_precision_recall_f1(
                        y_preds, y_true)
                    # print stat
                    micro_stats_group_rest.append(stat)
                    micro_stats_inst_rest.append(stat)
            # Compute micro-averaged metrics per group
            prec, rec, f1 = compute_micro_metrics(micro_stats_group_learn)
            print "\n##############################################"
            print "{} learn: prec: {} rec: {} f1: {}".format(group, prec, rec,
                                                             f1)
            all_stats[inst]["learn"][group] = [prec, rec, f1]
            prec, rec, f1 = compute_micro_metrics(micro_stats_group_rest)
            print "{} rest: prec: {} rec: {} f1: {}".format(group, prec, rec,
                                                            f1)
            print "##############################################\n"
            all_stats[inst]["rest"][group] = [prec, rec, f1]
        # Compute micro-averaged metrics per institution
        print "\n##############################################"
        prec, rec, f1 = compute_micro_metrics(micro_stats_inst_learn)
        print "{} learn: prec: {} rec: {} f1: {}".format(inst, prec, rec, f1)
        all_stats[inst]["learn"]["All"] = [prec, rec, f1]
        prec, rec, f1 = compute_micro_metrics(micro_stats_inst_rest)
        print "{} rest: prec: {} rec: {} f1: {}".format(inst, prec, rec, f1)
        print "##############################################\n"
        all_stats[inst]["rest"]["All"] = [prec, rec, f1]
    return all_stats


def compute_metric(metrics, idx):
    """
    Aggregates a specific metric over all annotators.

    Parameters
    ----------
    metrics: List of lists of float - each inner list represents 8 metrics for
    for a single annotator, the first 4 from learning phase, the other 4 from
    rest.
    idx: index of metric that should be aggragated.

    Returns
    -------
    float.
    Overall score w.r.t. a metric.

    """
    measurements = len(metrics)
    total = 0
    for anno in metrics:
        total += anno[idx]
    return 1.0*total / measurements


def extract_labels_text(interval):
    """
    Extracts from the labels and tweet messages in a given interval the separate
    labels and tweet texts.

    Parameters
    ----------
    interval: list of tuples: each tuple contains the labels and at the end the
    tweet message. The i-th tuple represents the labels of the annotator given
    to the i-th tweet.

    Returns
    -------
    List of str, list of str, list of str, list of str.
    Labels assigned for 1st set, labels assigned for 2nd set, labels assigned
    for 3rd set, tweet message.

    """
    firsts = []
    seconds = []
    thirds = []
    texts = []
    for tweet in interval:
        firsts.append(tweet[0])
        seconds.append(tweet[1])
        thirds.append(tweet[2])
        texts.append(tweet[3])
    return firsts, seconds, thirds, texts


def collect_data_per_annotator_per_group(tweet_path, anno_path, s, m_fatigue,
                                         cleaned=False):
    """
    Collects annotation labels per tweet  per annotator per group in an
    institution. If a label was missing, a dummy value, NA (=" "), is assigned
    instead.

    Parameters
    ----------
    tweet_path: str - path to tweet csv dataset.
    anno_path: str - path to annotator csv dataset.
    s: list of tuples - interval for group S. Each tuple describes the interval
    boundaries.
    m_fatigue: list of tuples - interval for group M with fatigue. Each
    tuple describes the interval boundaries. Boundaries for M without fatigue
    are the same, only the 2nd last interval is merged with the last one.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing average
    annotation times.

    Returns
    -------
    Dict.
    Key are the annotator groups "S", "M", "L" and values are lists of lists of
    lists of floats, assuming that they should be considered.
    Each inner list represents all annotation labels of a single annotator for
    a specific tweetin the
    respective group.
    Hence, the len(outer list) = number of annotators in a group per
    institution.

    """
    # Group is key, inner data structure:
    # List of annotators; list of intervals per annotator; list of 3 labels per
    # tweet in an interval (+ tweet text)
    # [[[label1 for tid1, label2 for tid1, label3 for tid1, text of tid1],...]
    # ...]
    annos = {"S": {},
             "M": {},
             # "L": []
             }
    print "intervals: S: [{}-{}) [{}-{}) M_fatigue: [{}-{}) " \
          "[{}-{}) [{}-{}) M_no_fatigue: [{}-{}) [{}-{})"\
          .format(s[0], s[1], s[1], s[2], m_fatigue[0], m_fatigue[1],
                  m_fatigue[1], m_fatigue[2], m_fatigue[2],
                  m_fatigue[3], m_fatigue[0], m_fatigue[1],
                  m_fatigue[1], m_fatigue[3])
    data = pseudo_db.Data(tweet_path, anno_path)
    # For each annotator
    for anno in data.annotators.all_annos():
        # All labels assigned by anno
        anno_labels = []
        group = anno.get_group()
        if group != "L":
            username = anno.get_name()
            # print "Collect tweets for annotator '{}'".format(username)
            # Lists in which annotator's tweets for the respective
            # interval are
            # stored: 2 interval for S, 2 interval for M without fatigue, 3
            # intervals for M with fatigue
            s_learn_tweets = []
            s_rest_tweets = []
            # This interval is the same in M with/without fatigue
            m_learn_tweets = []
            m_rest_tweets = []
            m_fat_tweets = []
            m_fat_rest_tweets = []
            # print "{} in {}:".format(username, group)

            # For each tweet
            for idx, t in enumerate(anno.get_labeled_tweets()):
                # print "#tweet:", idx
                # Add labels + tweet text at end
                labels = t.get_anno_labels()
                tweet_labels = []
                text = t.get_text()
                rel_label = labels[0]
                # First level
                tweet_labels.append(rel_label)
                # Second and third level might not have been assigned
                l2 = pseudo_db.NA
                l3 = pseudo_db.NA
                # Discard remaining labels if annotator chose "Irrelevant"
                # Consider other sets of labels only iff either the cleaned
                # dataset should be created and the label is "relevant" OR
                # the raw dataset should be used.
                if (cleaned and rel_label != "Irrelevant") or not cleaned:
                    # Second level
                    l2 = labels[1]
                    # Annotator labeled the 3rd set of labels as well
                    if l2 == pseudo_db.GROUND_TRUTH["Non-factual"]:
                        # Third level
                        l3 = labels[2]
                        # tweet_labels.append(l3)
                tweet_labels.append(l2)
                tweet_labels.append(l3)
                # Add tweet text at end
                tweet_labels.append(text)
                # Add tweet to respective interval
                if group == "S":
                    # print "add to interval S"
                    # Learning phase
                    if idx < s[1]:
                        # print "add to learning phase"
                        s_learn_tweets.append(tweet_labels)
                    else:
                        # print "add to rest phase"
                        s_rest_tweets.append(tweet_labels)
                elif group == "M":
                    # print "add to M"
                    # Learning phase
                    if idx < m_fatigue[1]:
                        # print "add to learning phase"
                        m_learn_tweets.append(tweet_labels)
                    else:
                        # Rest
                        # print "add to M rest"
                        m_rest_tweets.append(tweet_labels)
                        if idx < m_fatigue[2]:
                            # print "add to M fatigue"
                            m_fat_tweets.append(tweet_labels)
                        else:
                            # print "add to M fatigue rest"
                            m_fat_rest_tweets.append(tweet_labels)
            # Add labels for interval (per tweet) to annotator
            if group == "S":
                anno_labels.append(s_learn_tweets)
                anno_labels.append(s_rest_tweets)
            if group == "M":
                # Then add M no fatigue intervals
                anno_labels.append(m_learn_tweets)
                anno_labels.append(m_rest_tweets)
                # Then add M fatigue intervals (learning phase is identical
                # with M no fatigue, so no need to add it again)
                anno_labels.append(m_fat_tweets)
                anno_labels.append(m_fat_rest_tweets)
                # Number of tweets in M no fatigue must be the same as
                # number of tweets in 2 intervals of M fatigue;
                assert(len(m_rest_tweets) == (len(m_fat_tweets) +
                                              len(m_fat_rest_tweets)))
            # print annos
            annos[group][username] = anno_labels
            # break
        # break
    return annos


def aggregate_labels_per_anno(y_preds, y_true):
    """
    Aggregate labels s.t. for each tweet of an annotator her 3 assigned labels
    are stored.

    Parameters
    ----------
    y_preds: list of list of str - inner lists represent predicted labels for
    next hierarchy level. The i-th entry in such a list is the i-th predicted
    label by a specific annotator. Outer list represents the different hierarchy
    levels.
    y_true: list of list of str - same as y_preds, but it represents the true
    labels that were assigned by the annotator.

    Returns
    -------
    list of list of str, list of list of str.
    Predicted labels for all tweets of annotator over all hierarchical levels.
    Inner lists represent predicted labels (each entry is for the next deeper
    hierarchy level)  assigned to a specific
    tweet. Outer list represents predictions for all tweets of a certain
    annotator.Second returned value represents ground truth labels:
    same as first returned value, but it represents the true
    labels that were assigned by the annotator.

    Raises
    ------
    ValueError if the number of predicted and true labels differ on any
    hierarchy level.
    ValueError if there's a different number of hierarchy levels in predicted
    and true labels.

    """
    y_p = []
    y_t = []
    if len(y_preds) == len(y_true) and len(y_preds) > 0:
        # Check that in each level we have the same number of predicted labels
        # and true labels because otherwise we can't guarantee that the i-th
        # entry in a certain level corresponds to the i-th tweet labeled at that
        # hierarchy level by a certain annotator
        for pred, true in zip(y_preds, y_true):
            if len(pred) != len(true):
                raise ValueError("There isn't the same number of predicted "
                                 "and true labels on all hierarchy levels!")
    else:
        raise ValueError("Different number of hierarchy levels for predicted"
                         "and true labels!")

    # There is the same number of labels available on each hierarchy level,
    # so we can iterate over each i-th tweet of a given annotator
    for i in range(len(y_preds[0])):
        # Get predicted labels of annotator
        y_p1, y_p2, y_p3 = y_preds[0][i], y_preds[1][i], y_preds[2][i]
        # Get true labels of annotator
        y_t1, y_t2, y_t3 = y_true[0][i], y_true[1][i], y_true[2][i]
        y_p.append([y_p1, y_p2, y_p3])
        y_t.append([y_t1, y_t2, y_t3])
    return y_p, y_t


def plot(src, dst, metric_name, group, similarity):
    """
    Plots a performance metric.

    Parameters
    ----------
    src: str - file to input csv file.
    dst: str - location where plot will be stored.
    metric_name: str - name of the hierarchical metric: "prec" precision,
    "rec" for recall, or "f1" for F1-score
    group: str - annotator group that should be plotted: "All" for
    institution, "S" for group S, "M" for group M
    similarity: str - name of the metric to be displayed: "subsequence" for
    longest common subsequence, "substring" for longest common substring,
    "edit" for edit distance.

    """
    x_learn, y_learn, x_rest, y_rest = \
        read_data_from_csv(src, metric_name, group, similarity)
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    ax.plot(x_learn, y_learn, color="blue", label="$LEARN$")
    ax.plot(x_rest, y_rest, color="red", label="$EXPLOIT$")
    # Add a dashed line
    x = (1, x_learn[-1])
    y_val = max(y_rest[-1], y_learn[-1]) + 0.01
    y = (y_val, y_val)
    ax.plot(x, y, "k--")
    # Hide the right and top spines (lines)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # Set labels of axes
    ax.set_xlabel("k")
    ax.set_ylabel("F1-score")
    # Add legend outside of plot
    legend = ax.legend(loc="lower right", shadow=True, fontsize=FONTSIZE)
    plt.xlim(1, x_rest[-1]+0.5)
    plt.ylim(0, 1)
    plt.savefig(dst, bbox_inches='tight', dpi=600)
    plt.close()


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))

    # Path to the tweets of MD
    md_tweets = os.path.join(base_dir, "export",
                             "tweets_hierarchical_md.csv")
    # Path to the annotators of MD
    md_annos = os.path.join(base_dir, "export",
                            "annotators_hierarchical_md.csv")
    # Path to the tweets of SU
    su_tweets = os.path.join(base_dir, "export",
                             "tweets_hierarchical_su.csv")
    # Path to the annotators of SU
    su_annos = os.path.join(base_dir, "export",
                            "annotators_hierarchical_su.csv")

    # Directory in which stats will be stored
    STAT_DIR = os.path.join(base_dir, "results", "stats", "label_reliability")
    FIG_DIR = os.path.join(base_dir, "results", "figures", "label_reliability")
    if not os.path.exists(STAT_DIR):
        os.makedirs(STAT_DIR)
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    # Number of tweets to see before learning phase in S is completed
    k_s = 20
    # Number of tweets to see before learning phase in M is completed
    k_m = 40
    s_interval = (0, k_s, 50)
    m_interval_with_fatigue = (0, k_m, 80, 150)
    # Maximum number of neighbors to consider for predicting labels of unknown
    # tweets
    neighbors = 10
    intervals = [s_interval, m_interval_with_fatigue]
    run_simulation(intervals, neighbors, k_s, k_m, STAT_DIR, md_tweets,
                   md_annos, su_tweets, su_annos, cleaned=False, keep_na=True)
    run_simulation(intervals, neighbors, k_s, k_m, STAT_DIR, md_tweets,
                   md_annos, su_tweets, su_annos, cleaned=True, keep_na=True)

    # Create plots
    fname = "md_label_reliability_k_{}_k_s_20_k_m_40_cleaned.txt"\
        .format(neighbors)
    src = os.path.join(STAT_DIR, fname)
    # fname = "md_label_reliability_k_10_k_s_20_k_m_40_f1_cleaned.pdf"
    fname = "md_substring_cleaned.pdf"
    dst = os.path.join(FIG_DIR, fname)
    plot(src, dst, "f1", "All", "substring")

    fname = "md_subsequence_cleaned.pdf"
    dst = os.path.join(FIG_DIR, fname)
    plot(src, dst, "f1", "All", "subsequence")

    fname = "md_edit_cleaned.pdf"
    dst = os.path.join(FIG_DIR, fname)
    plot(src, dst, "f1", "All", "edit")

    fname = "su_label_reliability_k_{}_k_s_20_k_m_40_cleaned.txt"\
        .format(neighbors)
    src = os.path.join(STAT_DIR, fname)
    # fname = "su_label_reliability_k_10_k_s_20_k_m_40_f1_cleaned.pdf"
    fname = "su_substring_cleaned.pdf"
    dst = os.path.join(FIG_DIR, fname)
    plot(src, dst, "f1", "All", "substring")

    fname = "su_subsequence_cleaned.pdf"
    dst = os.path.join(FIG_DIR, fname)
    plot(src, dst, "f1", "All", "subsequence")

    fname = "su_edit_cleaned.pdf"
    dst = os.path.join(FIG_DIR, fname)
    plot(src, dst, "f1", "All", "edit")
