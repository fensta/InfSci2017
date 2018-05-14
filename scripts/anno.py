"""
Implements a wrapper class for a single annotator.

IMPORTANT:
----------
In MD (M) there was a small bug, s.t. 3 annotators labeled the same tweet
twice. Hence, this class doesn't allow adding the 2nd annotation of that tweet,
so these annotators labeled only 149 instead of 150 tweets.
The following annotators are affected:

"""
from copy import deepcopy
import warnings


# Indicate that an entry wasn't found
NA = -1


class Tweets(object):
    """
    Wrapper class to store data about tweets.
    """
    def __init__(self):
        # General idea: all data of i-th tweet is stored at i-th position
        # in respective lists, hence we also store for each tweet ID its
        # index to access a tweet's data quickly
        # {tweet_id: idx}
        self.lookup = {}
        # Index of next tweet to be added
        self.idx = 0
        # List of tweet IDs
        self.ids = []
        # List of user IDs, who wrote tweets - same order as <ids>
        self.user_ids = []
        # List of separate annotation times per hierarchy level per tweet -
        # same order as <ids>
        self.anno_times_list = []
        # List of separate confidence times per hierarchy level per tweet -
        # same order as <ids>
        self.conf_times_list = []
        # List of annotation times per tweet - same order as <ids>
        self.anno_times = []
        # List of confidence times per tweet - same order as <ids>
        self.conf_times = []
        # List of lists: inner list represents labels assigned to i-th tweet
        self.labels = []
        # List of lists: inner list represents confidence labels assigned to
        # i-th tweet
        self.conf_labels = []
        # List of actual tweet messages
        self.texts = []
        # Classifier certainties for the predicted tweet labels using the
        # following format:
        # [
        #     # Labels for tweet 1 over for all hierarchy levels
        #     [
        #         # Labels for hierarchy level 1
        #         [
        #             # Tuples of label and respective classifier certainty
        #             [(LABEL1, certainty1), (LABEL2, certainty2)]
        #         ],
        #         # Labels for hierarchy level 2
        #         [
        #             # Tuples of label and respective classifier certainty
        #             [(LABEL3, certainty3), (LABEL4, certainty4)]
        #         ],
        #         # Labels for hierarchy level 3
        #         [
        #             # Tuples of label and respective classifier certainty
        #             [(LABEL5, certainty5), (LABEL6, certainty6)]
        #         ]
        #     ],
        #     # Tweet 2...
        # ]
        self.predictions = []

    def add_tweet(self, tid, uid, anno_time, anno_times, labels, text,
                  conf_labels, conf_time, conf_times):
        """
        Add a tweet for a user.

        Parameters
        ----------
        tid: str - tweet ID.
        uid: str - user ID (=author of tweet).
        anno_time: float - total annotation time.
        anno_times: list - separate annotation times per hierarchy level.
        labels: list of str - each string represents a label - the i-th string
        is the label for the i-th hierarchy level.
        text: str - actual tweet message.
        conf_time: float - total confidence time.
        conf_times: list - separate confidence times per hierarchy level.
        conf_labels: list of str - each string represents a label - the i-th
        confidence label

        """
        self.lookup[tid] = self.idx
        self.idx += 1
        self.user_ids.append(uid)
        self.anno_times.append(anno_time)
        self.conf_times.append(conf_time)
        self.anno_times_list.append(anno_times)
        self.conf_times_list.append(conf_times)
        self.labels.append(labels)
        self.conf_labels.append(conf_labels)
        self.ids.append(tid)
        self.texts.append(text)
        self.predictions.append([])

    def set_prediction(self, certain, tid):
        """
        Sets the classifier certainty (+ label) for a given tweet.

        Parameters
        ----------
        certain: float - classifier certainty.

        """
        if tid in self.lookup:
            idx = self.lookup[tid]
            self.predictions[idx].extend(certain)

    def get_anno_time(self, tid):
        """
        Gets the annotation time for a specific tweet.

        Parameters
        ----------
        tid: str - tweet ID.

        Returns
        -------
        float.
        Returns -1 if no entry exists.

        """
        if tid in self.lookup:
            idx = self.lookup[tid]
            return self.anno_times[idx]
        return NA

    def get_anno_times(self, tid):
        """
        Gets the individual annotation times per hierarchy level for a specific
        tweet. The i-th annotation time corresponds to the i-th level.

        Parameters
        ----------
        tid: str - tweet ID.

        Returns
        -------
        list of float.
        Returns -1 if no entry exists.

        """
        if tid in self.lookup:
            idx = self.lookup[tid]
            return self.anno_times_list[idx]
        return NA

    def exists(self, tid):
        """
        Returns if a tweet exists in the collection or not.

        Parameters
        ----------
        tid: str - tweet ID.

        Returns
        -------
        bool.
        True if it exists, else False.

        """
        if tid in self.lookup:
            return True
        return False

    def delete_tweet(self, tid):
        """
        Deletes a given tweet and its values and assumes it exists.

        Parameters
        ----------
        tid: str - tweet ID to be deleted.

        """
        idx = self.lookup[tid]
        del self.predictions[idx]
        del self.ids[idx]
        del self.labels[idx]
        del self.conf_labels[idx]
        del self.anno_times[idx]
        del self.conf_times[idx]
        del self.texts[idx]
        del self.anno_times_list[idx]
        del self.conf_times_list[idx]
        # self.lookup is rebuild hereafter, so no need to change it in here
        self.update_index()

    def update_index(self):
        """
        After deleting tweets, the indices must be updated correspondingly.
        """
        idx = 0
        self.lookup = {}
        # Rebuild lookup dictionary
        for tid in self.ids:
            self.lookup[tid] = idx
            idx += 1
        self.idx = idx

    def set_predictions(self, preds):
        if len(self.ids) == len(preds):
            self.predictions = preds

    def preds_to_dict(self):
        """
        Returns a dictionary representation of the predictions.

        Returns
        -------
        dict.
        {tweet ID:
            # Labels for tweet 1 over for all hierarchy levels
            [
                # Labels for hierarchy level 1
                [
                    # Tuples of label and respective classifier certainty
                    [(LABEL1, certainty1), (LABEL2, certainty2)]
                ],
                # Labels for hierarchy level 2
                [
                    # Tuples of label and respective classifier certainty
                    [(LABEL3, certainty3), (LABEL4, certainty4)]
                ],
                # Labels for hierarchy level 3
                [
                    # Tuples of label and respective classifier certainty
                    [(LABEL5, certainty5), (LABEL6, certainty6)]
                ]
            ],...
        }

        """
        preds = {}
        for idx, (tid, pred) in enumerate(zip(self.ids, self.predictions)):
            preds[tid] = pred
        return preds

    def labels_to_dict(self):
        """
        Returns a dictionary representation of the predictions.

        Returns
        -------
        dict.
        {tweet ID:
            # Labels for tweet 1 over for all hierarchy levels
            [
                # Labels for hierarchy level 1
                [
                    [LABEL1, LABEL2]
                ],
                # Labels for hierarchy level 2
                [
                    [LABEL3, LABEL4]
                ],
                # Labels for hierarchy level 3
                [
                    [LABEL5, LABEL6]
                ]
            ],...
        }

        """
        labels= {}
        for idx, (tid, pred) in enumerate(zip(self.ids, self.labels)):
            labels[tid] = pred
        return labels

    def __str__(self):
        return self.ids + " " + self.anno_times_list

    def __len__(self):
        return len(self.ids)


class Annotator(object):
    def __init__(self, username, group, wid="", ds_name=""):
        """

        Parameters
        ----------
        username: str - name of the annotator.
        group: str - annotator group of the annotator.
        wid: str - ID of worker for anonymization - otherwise it's not used.
        ds_name: str - name of dataset to which annotator belongs - only used
        for anonymization.

        """
        # Annotator name
        self.name = username
        # Annotator group
        self.group = group
        # Annotator ID - for anonymization
        self.wid = wid
        # Dataset name to which annotator belongs - each annotator labeled only
        # tweets in a single dataset
        self.ds_name = ds_name

        # Tweets
        # -------
        self.tweets = Tweets()

    def add_tweet(self, tid, anno_time, labels, text, uid="", anno_times=[],
                  conf_labels=[], conf_time=0, conf_times=[]):
        """
        Add a tweet for a user.

        Parameters
        ----------
        tid: str - tweet ID.
        anno_time: float - total annotation time.
        anno_times: list - separate annotation times per hierarchy level - only
        used for anonymization.
        labels: list of str - each string represents a label - the i-th string
        is the label for the i-th hierarchy level.
        text: str - actual tweet message.
        uid: str - user ID (= author of tweet) - only used for anonymization,
        so it can be set to an arbitrary value otherwise.
        conf_labels: list of str - each string represents a confidence label -
        the i-th label - only used for anonymization.
        conf_time: float - total confidence time - only used for anonymization.
        conf_times: list - separate confidece times per hierarchy level - only
        used for anonymization.

        """
        # Some annotators labeled the same tweet twice - probably due to an
        # undiscovered bug in the annotation tool, so don't add any tweet
        # a second time
        if tid not in self.tweets.lookup:
            self.tweets.add_tweet(tid, uid, anno_time, anno_times, labels,
                                  text, conf_labels, conf_time, conf_times)
        else:
            warnings.warn("{} labeled tweet {} before!".format(self.name, tid))

    def set_predictions(self, preds):
        """
        Sets the predicted labels (and probabilities) for tweets.

        Parameters
        ----------
        preds: list of predictions for each tweet in the early stage.

        """
        self.tweets.set_predictions(preds)

    def set_prediction(self, tid, pred):
        """
        Sets the predicted labels (and probabilities) for a tweet.

        Parameters
        ----------
        tid: str - tweet ID for which predictions should be set.
        pred: list of predictions for each tweet in the early stage.

        """
        if self.tweets.exists(tid):
            self.tweets.set_prediction(pred, tid)
        else:
            raise ValueError("Tweet with ID {} wasn't labeled by {}!"
                             .format(tid, self.name))

    def delete_tweets(self, tids):
        """
        Deletes the specified tweets.

        Parameters
        ----------
        tids: list of str - each string is a tweet ID that should be deleted.

        """
        for tid in tids:
            if tid in self.tweets.lookup:
                self.tweets.delete_tweet(tid)

    def get_texts(self):
        """Returns the tweet texts of an annotator's tweets"""
        return self.tweets.texts

    def get_labels_per_level(self):
        """
        Retrieves the labels of the tweets per hierarchy level.

        Returns
        -------
        List of lists of str.
        There are n inner lists if there are n hierarchy levels. At the i-th
        index in an inner list we find the label of the i-th tweet according to
        self.tweets.ids.

        """
        return zip(*self.tweets.labels)

    def get_labels(self):
        """
        Returns a dictionary representation of the labels for the tweets.
        """
        return self.tweets.labels_to_dict()

    def keep_k(self, k):
        """
        Given an Annotator object, it's modified s.t. it has only the first
        k tweets.

        k: int - number of tweets to keep. Note that it's assumed that k <
        len(self.tweets), i.e. that there are at least k tweets available.

        """
        # Keep only the first k tweets
        self.tweets.ids = self.tweets.ids[:k]
        self.tweets.texts = self.tweets.texts[:k]
        self.tweets.labels = self.tweets.labels[:k]
        self.tweets.anno_times = self.tweets.anno_times[:k]
        self.tweets.predictions = self.tweets.predictions[:k]
        self.tweets.update_index()

    def keep_rest(self, k):
        """
        Given an Annotator object, it's modified s.t. given that annotator
        labeled n tweets, only the last n-k tweets are kept. In other words,
        the first k tweets are discarded.

        k: int - number of tweets to skip. Note that it's assumed that k <
        len(self.tweets), i.e. that there are at least k tweets available.

        """
        # 2. Keep only the remaining n-k tweets
        self.tweets.ids = self.tweets.ids[k:]
        self.tweets.texts = self.tweets.texts[k:]
        self.tweets.labels = self.tweets.labels[k:]
        self.tweets.anno_times = self.tweets.anno_times[k:]
        self.tweets.predictions = self.tweets.predictions[k:]
        self.tweets.update_index()

    def get_predictions(self):
        """
        Returns a dictionary representation of the predictions for the tweets..
        """
        return self.tweets.preds_to_dict()

    #############
    # Iterators #
    #############
    def all_tweets(self):
        """
        Returns an iterator over all tweets of an annotator.

        """
        for tid in self.tweets.ids:
            yield tid

    def __str__(self):
        return "{} ({}) labeled tweets ({})"\
            .format(self.name, self.group, len(self.tweets))

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    # https://stackoverflow.com/questions/1500718/what-is-the-right-way-to-override-the-copy-deepcopy-operations-on-an-object-in-p
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def __len__(self):
        return len(self.tweets)


def print_predictions(preds, skip_empty=False):
    """
    Prints out the predictions.

    Parameters
    ----------
    preds: dict - predictions.
    skip_empty: bool - True if tweets without any predicted labels shouldn't be
    shown. Default: False.

    """
    for tid in preds:
        print "tweet", tid
        # Skip tweets without predicted labels
        if skip_empty:
            if len(preds[tid]) > 0:
                for i, level in enumerate(preds[tid]):
                    print "{} level:".format(i+1)
                    print level
        else:
            for i, level in enumerate(preds[tid]):
                print "{} level:".format(i+1)
                print level


if __name__ == "__main__":
    a = Annotator("A", "S")
    b = Annotator("B", "M")
    a.add_tweet("123", 5.5, [1, 3, 1.5], ["a", "b", "c"], "hi")
    a.add_tweet("987", 15.3, [6, 8.3, 1], ["a", "d", "e"], "yo")
    # Create a deep copy and test if both instances are truly independent
    c = deepcopy(a)
    c.add_tweet("555", 6, [1, 3, 2], ["a", "a", "a"], "boo")
    c.add_tweet("666", 7, [3, 3, 1], ["b", "b", "b"], "oof")
    labels = [[(u'Relevant', 1.0)],  # First level
              [(u'Non-factual', 1.0)],  # Second level
              [(u'Negative', 0.6), (u'Positive', 0.4)]]  # Third level
    c.set_prediction("123", labels)
    print a
    print b
    print c

    print "times"
    for times in a.tweets.anno_times_list:
        print times

    print "ITERATE OVER LIST REPRESENTATION OF PREDICTIONS:"
    print c.tweets.predictions

    # Examples on how you can iterate all the predicted labels of a tweet
    for idx, t in enumerate(c.tweets.predictions):
        print "labels for tweet", idx+1
        print "for tweet", t
        for i, level in enumerate(t):
            print "{} level:".format(i+1)
            print level
            # for tpl in level:
            #     print tpl

    preds = c.get_predictions()
    print "ITERATE OVER DICTIONARY REPRESENTATION OF PREDICTIONS:"
    print preds
    for tid in preds:
        print tid
        for i, level in enumerate(preds[tid]):
            print "{} level:".format(i+1)
            print level
            # for tpl in level:
            #     print tpl

    assert(len(c.tweets) == (len(a.tweets) + 2))
    print c.tweets.predictions
    print a.tweets.predictions
    assert(len(c.tweets.predictions) == (len(a.tweets.predictions)+2))

    # Use iterators
    print "iterate over all tweets"
    for tid in c.all_tweets():
        print tid

    # Have only 1 tweet in early stage and the rest in late stage
    d = deepcopy(c)
    d.keep_k(1)
    print "d", d
    assert(len(d.tweets) == 1 and d.tweets.ids[0] == "123" and len(c.tweets) ==
           4)

    labels = [
        # 1st tweet
        [
            [(u'Relevant', 1.0)],  # First level
            [(u'Non-factual', 1.0)],  # Second level
            [(u'Negative', 0.6), (u'Positive', 0.4)],  # Third level
        ],
        # 2nd tweet
        [
            [(u'Relevant', 1.0)],  # First level
            [(u'Factual', 1.0)],  # Second level
            [(u'', 0.6), (u'Positive', 0.4)],  # Third level
        ],
        # 3rd tweet
        [
            [(u'IrRelevant', 1.0)],  # First level
            [(u'', 1.0)],  # Second level
            [(u'', 0.6), (u'Positive', 0.4)]  # Third level
        ],
        # 4th tweet
        [
            [(u'IrRelevant', 1.0)],  # First level
            [(u'', 1.0)],  # Second level
            [(u'', 0.6), (u'Positive', 0.4)]  # Third level
        ]
    ]
    f = deepcopy(d)
    f.set_predictions(labels)
    preds = f.get_predictions()
    print "SET ALL PREDICTIONS OF LATE STAGE"
    print preds
    for idx, tid in enumerate(preds):
        # There are 2 predicted labels on level 3 for tweet 1
        if idx == 0:
            assert(len(preds[tid][2]) == 2)
        # In tweet 3, the predicted labels for level 3 are:
        # [(u'', 0.6), (u'Positive', 0.4)]
        if idx == 2:
            assert(preds[tid][2] == [(u'', 0.6), (u'Positive', 0.4)])
        print tid
        for i, level in enumerate(preds[tid]):
            print "{} level:".format(i+1)
            print level

    e = deepcopy(c)
    # c labeled 4, so we want to keep 4-1 tweets
    e.keep_rest(1)
    print "e", e
    assert(len(e.tweets) == 3 and e.tweets.ids[0] == "987")

    # Test deep copy
    old_len = len(c.tweets)
    # Delete a tweet
    c.delete_tweets(["123"])
    print "c", c
    assert(len(c.tweets) == old_len - 1)
    assert("123" not in c.tweets.lookup)
    for tid in c.all_tweets():
        print tid

    print f.tweets.labels
    print zip(*f.tweets.labels)
    print "predictions", f.get_labels_per_level()
    print "level 1:", f.get_labels_per_level()[0]
    print "level 2:", f.get_labels_per_level()[1]
    print "level 3:", f.get_labels_per_level()[2]

    # Deep copy of a list of annos
    l1 = [a, b, c]
    print "l1"
    for anno in l1:
        print anno
    print "l2"
    l2 = deepcopy(l1)
    for anno in l2:
        print anno
    l2[1].add_tweet("666", 7, [1, 2, 4], ["b", "b", "b"], "oof")
    print "after l2 was changed"
    for anno in l1:
        print anno
    print "l2"
    for anno in l2:
        print anno
    assert(len(l2[1].tweets) == len(l1[1].tweets) + 1)
