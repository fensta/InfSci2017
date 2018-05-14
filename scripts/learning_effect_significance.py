"""
Analyzes the median annotation times of annotators for her first k, the last k,
and a combination of both:
- per institution
- per group
- in total

Visualizes the the comparisons and performs paired significance tests for the
following null hypotheses:
NOTE THAT firstK and lastK are SETS to avoid having to control for ordering
effects in the tweets.

In the remainder, the abstract H0 are listed, the used significance tests, and
the specific H0 derived from the abstract ones.
--------------------------------------------------------------------------------
1. Per group per institution:
a) H0: there is no difference in the firstK tweets among the annotators
b) H0: there is no difference in the lastK tweets among the annotators
c) H0: there is no difference between the set of firstK tweets and the set of
the lastK tweets

Tests:
1.H0 - 3.H0:
1-sample Mann-Whitney U, 1-sample Wilcoxon signed rank test,
1-sample unpaired t-test(Baum-Welch)
4.H0 - 6.H0 (unpaired):
2-sample Mann-Whitney U, 2-sample Wilcoxon-Mann-Whitney rank sum test,
2-sample unpaired t-test(Baum-Welch)

Unpaired because we shuffled firstk and lastk to avoid having to control the
order of the tweets, which means a loss in power of the test, but if results
are still significant, they'll also be significant with paired test

1.H0: there is no difference in the firstK tweets among the annotators of
<GROUP> in <INSTITUTION>.
2.H0: there is no difference in the lastK tweets among the annotators of
<GROUP> in <INSTITUTION>.
3.H0: there is no difference in the firstk+lastK tweets among the annotators of
<GROUP> in <INSTITUTION>.
4.H0: there is no difference in the set of the  median annotation times of the
firstk tweets compared with the set of the  lastk median annotation times in
tweets of <GROUP> in <INSTITUTION>.
5.H0: there is no difference in the set of the  median annotation times of the
firstk tweets compared with the set of the  firstk+lastk median annotation times
in tweets of <GROUP> in <INSTITUTION>.
6.H0: there is no difference in the set of the  median annotation times of the
lastk tweets compared with the set of the  firstk+lastk median annotation times
in tweets of <GROUP> in <INSTITUTION>.
--------------------------------------------------------------------------------
2. Using MD and SU together and compare the groups:
a) H0: there is no difference in the firstK tweets among the annotators
b) H0: there is no difference in the lastK tweets among the annotators
c) H0: there is no difference between the set of firstK tweets and the set of
the lastK tweets

Tests:
1.H0 - 3.H0:
1-sample Mann-Whitney U, 1-sample Wilcoxon signed rank test,
1-sample unpaired t-test(Baum-Welch)
4.H0 - 6.H0 (unpaired):
2-sample Mann-Whitney U, 2-sample Wilcoxon-Mann-Whitney rank sum test,
2-sample unpaired t-test(Baum-Welch)

Unpaired because we shuffled firstk and lastk to avoid having to control the
order of the tweets, which means a loss in power of the test, but if results
are still significant, they'll also be significant with paired test

1.H0: there is no difference in the firstK tweets among the annotators in
<GROUP>.
2.H0: there is no difference in the lastK tweets among the annotators in
<GROUP>.
3.H0: there is no difference between the set of firstK tweets and the set of
the lastK tweets in <GROUP>.
4.H0: there is no difference in the set of the  median annotation times of the
firstk tweets compared with the set of the  lastk median annotation times in
tweets.
5.H0: there is no difference in the set of the median annotation times of the
firstk tweets compared with the set of the  firstk+lastk median annotation
times in tweets.
6.H0: there is no difference in the set of the median annotation times of the
lastk tweets compared with the set of the firstk+lastk median annotation times
in tweets.
--------------------------------------------------------------------------------
3.Between institutions, compare the groups:
a) H0: there is no difference between MD and SU for the firstK tweets
b) H0: there is no difference between MD and SU for the lastK tweets
c) H0: there is no difference between the firstK+lastK tweets of MD and the
firstK+lastK tweets of SU

Tests:
1.H0-6.H0:
2-sample Mann-Whitney U, 2-sample Wilcoxon-Mann-Whitney rank sum test,
2-sample unpaired t-test(Baum-Welch)

Unpaired because we shuffled firstk and lastk to avoid having to control the
order of the tweets, which means a loss in power of the test, but if results
are still significant, they'll also be significant with paired test

# MD S vs. SU S
1.H0: there is no difference in the set of firstK tweets among the annotators of S in SU and MD.
2.H0: there is no difference in the set of firstK tweets among the annotators of S in SU and the set of lastK tweets in MD in S.
3.H0: there is no difference in the set of firstK tweets among the annotators of S in SU and the set of firstK+lastK tweets in MD in S.
4.H0: there is no difference in the set of lastK tweets among the annotators of S in SU and the set of firstK tweets in MD in S.
5.H0: there is no difference in the set of lastK tweets among the annotators of S in SU and MD.
6.H0: there is no difference in the set of lastK tweets among the annotators of S in SU and the set of firstK+lastK tweets in MD in S.
7.H0: there is no difference in the set of firstk+lastK tweets among the annotators of S in SU and the set of firstK tweets in MD in S.
8.H0: there is no difference in the set of firstk+lastK tweets among the annotators of S in SU and the set of lastK tweets in MD in S.
9.H0: there is no difference in the set of firstk+lastK tweets among the annotators of S in SU and MD.
# MD M vs. SU M
10.H0: there is no difference in the set of firstK tweets among the annotators of M in SU and MD.
11.H0: there is no difference in the set of firstK tweets among the annotators of M in SU and the set of lastK tweets in MD in M.
12.H0: there is no difference in the set of firstK tweets among the annotators of M in SU and the set of firstK+lastK tweets in MD in M.
13.H0: there is no difference in the set of lastK tweets among the annotators of M in SU and the set of firstK tweets in MD in M.
14.H0: there is no difference in the set of lastK tweets among the annotators of M in SU and MD.
15.H0: there is no difference in the set of lastK tweets among the annotators of M in SU and the set of firstK+lastK tweets in MD in M.
16.H0: there is no difference in the set of firstk+lastK tweets among the annotators of M in SU and the set of firstK tweets in MD in M.
17.H0: there is no difference in the set of firstk+lastK tweets among the annotators of M in SU and the set of lastK tweets in MD in M.
18.H0: there is no difference in the set of firstk+lastK tweets among the annotators of M in SU and MD.
# MD S vs. SU M
19.H0: there is no difference in the set of firstK tweets of S in MD compared with the set of firstK tweets of the annotators of M in SU.
20.H0: there is no difference in the set of firstK tweets of S in MD compared with the set of lastK tweets of the annotators of M in SU.
21.H0: there is no difference in the set of firstK tweets of S in MD compared with the set of firstK+lastK tweets of the annotators of M in SU.
22.H0: there is no difference in the set of lastK tweets of S in MD compared with the set of firstK tweets of the annotators of M in SU.
23.H0: there is no difference in the set of lastK tweets of S in MD compared with the set of lastK tweets of the annotators of M in SU.
24.H0: there is no difference in the set of lastK tweets of S in MD compared with the set of firstK+lastK tweets of the annotators of M in SU.
25.H0: there is no difference in the set of firstK+lastK tweets of S in MD compared with the set of firstk tweets of the annotators of M in SU.
26.H0: there is no difference in the set of firstK+lastK tweets of S in MD compared with the set of lastK tweets of the annotators of M in SU.
27.H0: there is no difference in the set of firstK+lastK tweets of S in MD compared with the set of firstk+lastK tweets of the annotators of M in SU.
# MD M vs. SU S
28.H0: there is no difference in the set of firstK tweets of M in MD compared with the set of firstK tweets of the annotators of S in SU.
29.H0: there is no difference in the set of firstK tweets of M in MD compared with the set of lastK tweets of the annotators of S in SU.
30.H0: there is no difference in the set of firstK tweets of M in MD compared with the set of firstK+lastK tweets of the annotators of S in SU.
31.H0: there is no difference in the set of lastK tweets of M in MD compared with the set of firstK tweets of the annotators of S in SU.
32.H0: there is no difference in the set of lastK tweets of M in MD compared with the set of lastK tweets of the annotators of S in SU.
33.H0: there is no difference in the set of lastK tweets of M in MD compared with the set of firstK+lastK tweets of the annotators of S in SU.
34.H0: there is no difference in the set of firstK+lastK tweets of M in MD compared with the set of firstk tweets of the annotators of S in SU.
35.H0: there is no difference in the set of firstK+lastK tweets of M in MD compared with the set of lastK tweets of the annotators of S in SU.
36.H0: there is no difference in the set of firstK+lastK tweets of M in MD compared with the set of firstk+lastK tweets of the annotators of S in SU.
# MD S vs. MD M
37.H0: there is no difference in the set of firstK tweets of M in MD compared with the set of firstK tweets of the annotators of S in MD.
38.H0: there is no difference in the set of firstK tweets of M in MD compared with the set of lastK tweets of the annotators of S in MD.
39.H0: there is no difference in the set of firstK tweets of M in MD compared with the set of firstK+lastK tweets of the annotators of S in MD.
40.H0: there is no difference in the set of lastK tweets of M in MD compared with the set of firstK tweets of the annotators of S in MD.
41.H0: there is no difference in the set of lastK tweets of M in MD compared with the set of lastK tweets of the annotators of S in MD.
42.H0: there is no difference in the set of lastK tweets of M in MD compared with the set of firstK+lastK tweets of the annotators of S in MD.
43.H0: there is no difference in the set of firstK+lastK tweets of M in MD compared with the set of firstk tweets of the annotators of S in MD.
44.H0: there is no difference in the set of firstK+lastK tweets of M in MD compared with the set of lastK tweets of the annotators of S in MD.
45.H0: there is no difference in the set of firstK+lastK tweets of M in MD compared with the set of firstk+lastK tweets of the annotators of S in MD.
# SU S vs. SU M
46.H0: there is no difference in the set of firstK tweets of M in SU compared with the set of firstK tweets of the annotators of S in SU.
47.H0: there is no difference in the set of firstK tweets of M in SU compared with the set of lastK tweets of the annotators of S in SU.
48.H0: there is no difference in the set of firstK tweets of M in SU compared with the set of firstK+lastK tweets of the annotators of S in SU.
49.H0: there is no difference in the set of lastK tweets of M in SU compared with the set of firstK tweets of the annotators of S in SU.
50.H0: there is no difference in the set of lastK tweets of M in SU compared with the set of lastK tweets of the annotators of S in SU.
51.H0: there is no difference in the set of lastK tweets of M in SU compared with the set of firstK+lastK tweets of the annotators of S in SU.
52.H0: there is no difference in the set of firstK+lastK tweets of M in SU compared with the set of firstk tweets of the annotators of S in SU.
53.H0: there is no difference in the set of firstK+lastK tweets of M in SU compared with the set of lastK tweets of the annotators of S in SU.
54.H0: there is no difference in the set of firstK+lastK tweets of M in SU compared with the set of firstk+lastK tweets of the annotators of S in SU.

"""
import os
import subprocess

import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import pseudo_db


# For Information sciences
FONTSIZE = 12
plt.rcParams.update({'font.size': FONTSIZE})


def compute_learning_effect_significance(
        script_dir, fig_dir, stat_dir, md_tw_p, md_a_p, su_tw_p, su_a_p,
        cleaned=False, n=50, with_l=False):
    """
    Computes significances for the different hypotheses and creates 1 plot
    per hypothesis.

    NOTE: with_l isn't supported at the moment in R!!!!
    https://gist.github.com/mblondel/1761714 shows how to do 1-sample Wilcoxon
    in Python...(saw it too late).

    Parameters
    ----------
    script_dir: str - directory in which the scripts, especially the R scripts
    are stored.
    fig_dir: str - directory in which the plots will be stored.
    stat_dir: str - directory in which the stats will be stored.
    md_tw_p: str - path to MD tweets dataset in csv format.
    md_a_p: str - path to MD annotators dataset in csv format.
    su_tw_p: str - path to SU tweets dataset in csv format.
    su_a_p: str - path to SU annotators dataset in csv format.
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing average
    annotation times.
    n: int - maximum number of tweets that should be used from the
    beginning and end of the annotation times of an annotator to compute
    the respective median annotation time.
    with_l: True if annotators of L should also be used.

    """
    if with_l:
        raise NotImplementedError("Not implemented in R!")

    dataset_type = "raw"
    if cleaned:
        dataset_type = "cleaned"
    size = "without_l"
    if with_l:
        size = "with_l"
    # Each inner list represents annotation times of a single annotator
    # {institution: {group: [[1, 3], [4, 6, 7, 2]]}}
    times_group = {
        "md": get_total_anno_times(md_tw_p, md_a_p, cleaned=cleaned,
                                   with_l=with_l),
        "su": get_total_anno_times(su_tw_p, su_a_p, cleaned=cleaned,
                                   with_l=with_l)
    }
    if with_l:
        print "md groups:", len(times_group["md"]["S"]) + \
                            len(times_group["md"]["M"]) + \
                            len(times_group["md"]["L"])
        print "su groups:", len(times_group["su"]["S"]) + \
                            len(times_group["su"]["M"]) + \
                            len(times_group["su"]["L"])
    else:
        print "md groups:", len(times_group["md"]["S"]) + \
                            len(times_group["md"]["M"])
        print "su groups:", len(times_group["su"]["S"]) + \
                            len(times_group["su"]["M"])

    # Structure annotation times for H0 1a-c and H0 3a-c
    md_s = np.array(times_group["md"]["S"], dtype=float)
    md_m = np.array(times_group["md"]["M"], dtype=float)
    su_s = np.array(times_group["su"]["S"], dtype=float)
    su_m = np.array(times_group["su"]["M"], dtype=float)

    print "md_s", md_s.shape
    print "md_m", md_m.shape
    print "su_s", su_s.shape
    print "su_m", su_m.shape

    md_l = []
    su_l = []

    # If L should also be tested
    if with_l:
        md_l = np.array(times_group["md"]["L"], dtype=float)
        su_l = np.array(times_group["su"]["L"], dtype=float)
        print "md_l", md_l.shape
        print "su_l", su_l.shape

    # Structure annotation times for H0 2a-c
    times_inst = {
        "S": [],
        "M": [],
        "L": []
    }
    for inst, groups in times_group.iteritems():
        for group in groups:
            # Discard L if desired
            if (group == "L" and with_l) or group != "L" or not with_l:
                times_inst[group].extend(times_group[inst][group])
    print "S all inst:", len(times_inst["S"])
    print "M all inst:", len(times_inst["M"])
    print "L all inst:", len(times_inst["L"])
    s = np.array(times_inst["S"], dtype=float)
    m = np.array(times_inst["M"], dtype=float)
    print "S", s.shape
    print "M", m.shape
    if with_l:
        l = np.array(times_inst["L"], dtype=float)
        print "L", l.shape

    #############################################
    # Store datasets for R for H0 1a-c and 3a-c #
    #############################################
    #  1. MD_S
    fname = "input_md_s_learning_effect_groups_in_same_institution_{}_{}.txt"\
        .format(size, dataset_type)
    mds_fpath = os.path.join(stat_dir, fname)
    md_dst_path = os.path.join(stat_dir,
                               "md_learning_effect_groups_in_same_institution"
                               "_{}_{}.txt".format(size, dataset_type))
    np.savetxt(mds_fpath, md_s, delimiter=",")

    # 2. MD_M
    fname = "input_md_m_learning_effect_groups_in_same_institution_{}_{}.txt"\
        .format(size, dataset_type)
    mdm_fpath = os.path.join(stat_dir, fname)
    # mdm_dst_path = stat_dir + \
    #     "md_m_learning_effect_groups_in_same_institution_{}_{}.txt"\
    #     .format(size, dataset_type)
    np.savetxt(mdm_fpath, md_m, delimiter=",")

    # 3. SU_S
    fname = "input_su_s_learning_effect_groups_in_same_institution_{}_{}.txt"\
        .format(size, dataset_type)
    sus_fpath = os.path.join(stat_dir, fname)
    su_dst_path = os.path.join(stat_dir,
                               "su_learning_effect_groups_in_same_institution"
                               "_{}_{}.txt".format(size, dataset_type))
    np.savetxt(sus_fpath, su_s, delimiter=",")

    # 4. SU_M
    fname = "input_su_m_learning_effect_groups_in_same_institution_{}_{}.txt"\
        .format(size, dataset_type)
    sum_fpath = os.path.join(stat_dir, fname)
    # sum_dst_path = stat_dir + \
    #     "su_m_learning_effect_groups_in_same_institution_{}_{}.txt"\
    #     .format(size, dataset_type)
    np.savetxt(sum_fpath, su_m, delimiter=",")

    # If L should also be tested
    if with_l:
        # 5. MD_L
        fname = "input_md_l_learning_effect_groups_in_same_institution_{}_" \
                "{}.txt".format(size, dataset_type)
        mdl_fpath = os.path.join(stat_dir, fname)
        mdl_dst_path = os.path.join(stat_dir,
                                    "md_l_learning_effect_groups_in_same_ins"
                                    "titution_{}_{}.txt"
                                    .format(size, dataset_type))
        np.savetxt(mdl_fpath, md_l, delimiter=",")

        # 6. SU_L
        fname = "input_su_l_learning_effect_groups_in_same_institution_{}_" \
                "{}.txt".format(size, dataset_type)
        sul_fpath = os.path.join(stat_dir, fname)
        sul_dst_path = os.path.join(stat_dir,
                                    "su_l_learning_effect_groups_in_same_insti"
                                    "tution_{}_{}.txt".format(size,
                                                              dataset_type))
        np.savetxt(sul_fpath, su_l, delimiter=",")

    ####################################
    # Store datasets for R for H0 2a-c #
    ####################################
    # 1. S
    fname = "input_s_learning_effect_groups_over_all_institutions_{}_{}.txt"\
        .format(size, dataset_type)
    s_fpath = os.path.join(stat_dir, fname)
    all_groups_s_dst_path = os.path.join(stat_dir,
                                         "s_learning_effect_groups_over_all_in"
                                         "stitutions_{}_{}.txt"
                                         .format(size, dataset_type))
    np.savetxt(s_fpath, s, delimiter=",")

    # 2. M
    fname = "input_m_learning_effect_groups_over_all_institutions_{}_{}.txt"\
        .format(size, dataset_type)
    m_fpath = os.path.join(stat_dir, fname)
    all_groups_m_dst_path = os.path.join(stat_dir,
                                         "m_learning_effect_groups_over_all_"
                                         "institutions_{}_{}.txt"
                                         .format(size, dataset_type))
    np.savetxt(m_fpath, m, delimiter=",")

    if with_l:
        # 3. L
        fname = "input_l_learning_effect_groups_over_all_institutions_{}_" \
                "{}.txt".format(size, dataset_type)
        l_fpath = os.path.join(stat_dir, fname)
        l_dst_path = os.path.join(stat_dir,
                                  "l_learning_effect_groups_over_all_instit"
                                  "utions_{}_{}.txt".format(size, dataset_type))
        np.savetxt(l_fpath, l, delimiter=",")

    #######################################
    # Create path names for H0 3a-c for R #
    #######################################
    #  1. S
    # fname1 = "input_md_s_learning_effect_groups_in_different_institutions_{}" \
    #         "_{}.txt".format(size, dataset_type)
    # fname2 = "input_su_s_learning_effect_groups_in_different_institutions_{}" \
    #         "_{}.txt".format(size, dataset_type)
    # mds1_fpath = stat_dir + fname1
    # sus1_fpath = stat_dir + fname2
    all_dst_path = os.path.join(stat_dir,
                                "learning_effect_groups_in_different_institu"
                                "tions_{}_{}.txt".format(size, dataset_type))

    # 2. M
    # fname1 = "input_md_m_learning_effect_groups_in_different_institutions_{}" \
    #          "_{}.txt".format(size, dataset_type)
    # fname2 = "input_su_m_learning_effect_groups_in_different_institutions_{}" \
    #          "_{}.txt".format(size, dataset_type)
    # mdm1_fpath = stat_dir + fname1
    # sum1_fpath = stat_dir + fname2
    # m1_dst_path = stat_dir + \
    #     "m_learning_effect_groups_in_different_institutions_{}_{}.txt"\
    #     .format(size, dataset_type)

    if with_l:
        # 3. L
        fname1 = "input_md_l_learning_effect_groups_in_different_institutions" \
                 "_{}_{}.txt".format(size, dataset_type)
        fname2 = "input_su_l_learning_effect_groups_in_different_institutions" \
                 "_{}_{}.txt".format(size, dataset_type)
        mdl1_fpath = os.path.join(stat_dir, fname1)
        sul1_fpath = os.path.join(stat_dir, fname2)
        l1_dst_path = os.path.join(stat_dir, "l_learning_effect_groups_in_diff"
                                             "erent_institutions_{}_{}.txt"
                                   .format(size, dataset_type))

    #############################################################
    # Feed parameters into R, so that it computes significances #
    #############################################################
    # Start R from Python:
    # http://www.kdnuggets.com/2015/10/integrating-python-r-executing-part2.html
    command = "Rscript"
    script_path = os.path.join(script_dir, "learning_effect_significance.R")
    ##########################################################################
    # a) Deal with intra institution group comparisons, i.e. null hypotheses #
    # 1a-1c                                                                  #
    ##########################################################################
    test_type = "intragroup"
    # 1. MD
    args = [test_type, str(n), mds_fpath, mdm_fpath, md_dst_path]
    cmd = [command, script_path] + args
    ending = "_{}_{}.pdf".format(size, dataset_type)
    try:
        success = subprocess.check_output(cmd, universal_newlines=True)
        if success:
            print "R script was run successfully for MD intra"
        plot_intra_group(md_dst_path, fig_dir, ending)
        plot_intra_group_paper(md_dst_path, fig_dir, ending)
    except subprocess.CalledProcessError, e:
        print "Error output:\n", e.output
    # 2. SU
    args = [test_type, str(n), sus_fpath, sum_fpath, su_dst_path]
    cmd = [command, script_path] + args
    try:
        success = subprocess.check_output(cmd, universal_newlines=True)
        if success:
            print "R script was run successfully for SU intra"
        plot_intra_group(su_dst_path, fig_dir, ending)
        plot_intra_group_paper(su_dst_path, fig_dir, ending)
    except subprocess.CalledProcessError, e:
        print "Error output:\n", e.output
    ####################################################################
    # b) Deal with group comparisons over all institutions, i.e. null  #
    # hypotheses 2a-2c                                                 #
    ####################################################################
    test_type = "overinstitutions"
    # 1. S
    args = [test_type, str(n), s_fpath, all_groups_s_dst_path]
    cmd = [command, script_path] + args
    ending = "_{}_{}.pdf".format(size, dataset_type)
    try:
        success = subprocess.check_output(cmd, universal_newlines=True)
        if success:
            print "R script was run successfully for S"
        plot_groups_over_insts(all_groups_s_dst_path, fig_dir, ending)
        plot_groups_over_insts_paper(all_groups_s_dst_path, fig_dir, ending)
    except subprocess.CalledProcessError, e:
        print "Error output:\n", e.output
    # 2. M
    args = [test_type, str(n), m_fpath, all_groups_m_dst_path]
    cmd = [command, script_path] + args
    try:
        success = subprocess.check_output(cmd, universal_newlines=True)
        if success:
            print "R script was run successfully for M"
        plot_groups_over_insts(all_groups_m_dst_path, fig_dir, ending)
        plot_groups_over_insts_paper(all_groups_m_dst_path, fig_dir, ending)
    except subprocess.CalledProcessError, e:
        print "Error output:\n", e.output

    # if with_l:
    #     # 3. L
    #     args = [l_fpath, str(n), l_dst_path, test_type]
    #     cmd = [command, script_path] + args
    #     success = subprocess.check_output(cmd, universal_newlines=True)
    #     if success:
    #         print "R script was run successfully for L"

    ####################################################################
    # c) Deal with group comparisons between institutions, i.e. null   #
    # hypotheses 3a-3c                                                 #
    ####################################################################
    test_type = "intergroup"

    # 1. S + M
    args = [test_type, str(n), mds_fpath, mdm_fpath, sus_fpath, sum_fpath,
            all_dst_path]
    cmd = [command, script_path] + args
    ending = "_{}_{}.pdf".format(size, dataset_type)
    try:
        success = subprocess.check_output(cmd, universal_newlines=True)
        if success:
            print "R script was run successfully for MD across S"

        plot_inter_groups(all_dst_path, fig_dir, ending)
        plot_inter_groups_paper(all_dst_path, fig_dir, ending)
    except subprocess.CalledProcessError, e:
        print "Error output:\n", e.output
    # if with_l:
    #     # 3. L
    #     args = [mdl1_fpath, str(n), l1_dst_path, test_type, sul1_fpath]
    #     cmd = [command, script_path] + args
    #     success = subprocess.check_output(cmd, universal_newlines=True)
    #     if success:
    #         print "R script was run successfully for MD across L"


def plot_intra_group(src, dst_dir, ending):
    """
    Reads the p-values from <src> and plots them in <dst_dir>.

    Parameters
    ----------
    src: str - path to txt file storing the p-values.
    dst_dir: str - path to directory where plots should be stored.
    ending: str - file ending for each plot.

    """
    if os.path.isfile(src):
        ##############
        # Parse file #
        ##############
        institution = os.path.basename(src).split("_")[0].upper()
        with open(src, "rb") as f:
            lines = f.readlines()
        is_body = False
        x = []
        # 3 significance tests per hypothesis in corresponding order of the
        # file
        # 1-sample tests
        tests1 = ["1-sample Mann-Whitney U", "1-sample Wilcoxon Signed Rank",
                  "1-sample Baum-Welch"]
        # 2-sample tests
        tests2 = ["2-sample Mann-Whitney U", "2-sample Wilcoxon rank sum",
                  "2-sample Baum-Welch"]
        ps = {}
        i = 0
        # Parse file
        for line in lines:
            # Remove \n
            line = line.rstrip()
            # Current hypothesis to which we add p-values; 1-based counting
            hypo = 0
            # Skip header and only read body
            if is_body:
                content = line.split(",")
                group = content[0].upper()
                # Skip group and k entry at beginning
                for idx, p in enumerate(content[2:]):
                    # Find out to which hypothesis the current p-value belongs
                    if idx % len(tests1) == 0:
                        hypo += 1
                    # Remove * or ** from significant entries
                    if p.endswith("**"):
                        p = p[:-2]
                    elif p.endswith("*"):
                        p = p[:-1]
                    p = float(p)
                    # Find out the test's name used for the idx-th entry -
                    # choose from 2-sample or 1-sample tests
                    if idx > 8:
                        test = tests2[idx % len(tests1)]
                    else:
                        test = tests1[idx % len(tests1)]
                    ps[hypo][test][group].append(p)
                    i += 1
            # Identify when body begins
            if line.startswith("++++++++++++++++++++++++++++++++"):
                is_body = True
            # Extract from header how many hypotheses exist
            if line.startswith("group k 1.H0"):
                # 3 tests per hypotheses, first 2 columns contain group and k
                hypos = (len(line.split(" ")) - 2) / len(tests2)
                print "hypos:", hypos
                # Initialize dict
                # {hypo1:
                #   {
                #       "Wilcoxon":
                #       {
                #           "S": {},
                #           "M": {}
                #       },
                #       "Mann":
                #       {
                #           "S": {},
                #           "M": {}
                #       },...
                #   },...
                # }
                for h in xrange(hypos):
                    ps[h+1] = {
                        "1-sample Wilcoxon Signed Rank": {
                                "S": [],
                                "M": []
                        },
                        "1-sample Mann-Whitney U": {
                                "S": [],
                                "M": []
                        },
                        "1-sample Baum-Welch": {
                                "S": [],
                                "M": []
                        },
                        "2-sample Baum-Welch": {
                                "S": [],
                                "M": []
                        },
                        "2-sample Wilcoxon rank sum": {
                                "S": [],
                                "M": []
                        },
                        "2-sample Mann-Whitney U": {
                                "S": [],
                                "M": []
                        }

                    }
        ############
        # Plotting #
        ############
        # 1 plot per group per hypothesis
        for hypo in ps:
            # S
            # Inner lists are y-values per significance test - it's name is in
            # <names_m>
            ys_s = []
            names_s = []
            # M
            names_m = []
            # Inner lists are y-values per significance test - it's name is in
            # <names_s>
            ys_m = []
            for test in ps[hypo]:
                # Plot For S
                ys_s.append(ps[hypo][test]["S"])
                names_s.append(test)
                # Plot for M
                ys_m.append(ps[hypo][test]["M"])
                names_m.append(test)

            _plot(hypo, "S", institution, ys_s, names_s, dst_dir, ending)
            _plot(hypo, "M", institution, ys_m, names_m, dst_dir, ending)
    else:
        raise IOError("{} doesn't exist".format(src))


def plot_intra_group_paper(src, dst_dir, ending):
    """
    Reads the p-values from <src> and plots them in <dst_dir>. Plotting is
    adjusted to the paper

    Parameters
    ----------
    src: str - path to txt file storing the p-values.
    dst_dir: str - path to directory where plots should be stored.
    ending: str - file ending for each plot.

    """
    if os.path.isfile(src):
        ##############
        # Parse file #
        ##############
        institution = os.path.basename(src).split("_")[0].upper()
        with open(src, "rb") as f:
            lines = f.readlines()
        is_body = False
        x = []
        # 3 significance tests per hypothesis in corresponding order of the
        # file
        # 1-sample tests
        tests1 = ["1-sample Mann-Whitney U", "1-sample Wilcoxon Signed Rank",
                  "1-sample Baum-Welch"]
        # 2-sample tests
        tests2 = ["2-sample Mann-Whitney U", "2-sample Wilcoxon rank sum",
                  "2-sample Baum-Welch"]
        ps = {}
        i = 0
        # Parse file
        for line in lines:
            # Remove \n
            line = line.rstrip()
            # Current hypothesis to which we add p-values; 1-based counting
            hypo = 0
            # Skip header and only read body
            if is_body:
                content = line.split(",")
                group = content[0].upper()
                # Skip group and k entry at beginning
                for idx, p in enumerate(content[2:]):
                    # Find out to which hypothesis the current p-value belongs
                    if idx % len(tests1) == 0:
                        hypo += 1
                    # Remove * or ** from significant entries
                    if p.endswith("**"):
                        p = p[:-2]
                    elif p.endswith("*"):
                        p = p[:-1]
                    p = float(p)
                    # Find out the test's name used for the idx-th entry -
                    # choose from 2-sample or 1-sample tests
                    if idx > 8:
                        test = tests2[idx % len(tests1)]
                    else:
                        test = tests1[idx % len(tests1)]
                    ps[hypo][test][group].append(p)
                    i += 1
            # Identify when body begins
            if line.startswith("++++++++++++++++++++++++++++++++"):
                is_body = True
            # Extract from header how many hypotheses exist
            if line.startswith("group k 1.H0"):
                # 3 tests per hypotheses, first 2 columns contain group and k
                hypos = (len(line.split(" ")) - 2) / len(tests2)
                print "hypos:", hypos
                # Initialize dict
                # {hypo1:
                #   {
                #       "Wilcoxon":
                #       {
                #           "S": {},
                #           "M": {}
                #       },
                #       "Mann":
                #       {
                #           "S": {},
                #           "M": {}
                #       },...
                #   },...
                # }
                for h in xrange(hypos):
                    ps[h+1] = {
                        "1-sample Wilcoxon Signed Rank": {
                                "S": [],
                                "M": []
                        },
                        "1-sample Mann-Whitney U": {
                                "S": [],
                                "M": []
                        },
                        "1-sample Baum-Welch": {
                                "S": [],
                                "M": []
                        },
                        "2-sample Baum-Welch": {
                                "S": [],
                                "M": []
                        },
                        "2-sample Wilcoxon rank sum": {
                                "S": [],
                                "M": []
                        },
                        "2-sample Mann-Whitney U": {
                                "S": [],
                                "M": []
                        }

                    }
        ############
        # Plotting #
        ############
        # 1 plot per group per hypothesis
        for hypo in ps:
            # S
            # Inner lists are y-values per significance test - it's name is in
            # <names_m>
            ys_s = []
            names_s = []
            # M
            names_m = []
            # Inner lists are y-values per significance test - it's name is in
            # <names_s>
            ys_m = []
            for test in ps[hypo]:
                if test == "2-sample Wilcoxon rank sum":
                    # Plot For S
                    ys_s.append(ps[hypo][test]["S"])
                    names_s.append(test)
                    # Plot for M
                    ys_m.append(ps[hypo][test]["M"])
                    names_m.append(test)

            _plot_paper(hypo, "S", institution, ys_s, names_s, dst_dir, ending)
            _plot_paper(hypo, "M", institution, ys_m, names_m, dst_dir, ending)
    else:
        raise IOError("{} doesn't exist".format(src))


def plot_groups_over_insts(src, dst_dir, ending):
    """
    Reads the p-values from <src> and plots them in <dst_dir>.

    Parameters
    ----------
    src: str - path to txt file storing the p-values.
    dst_dir: str - path to directory where plots should be stored.
    ending: str - file ending for each plot.

    """
    if os.path.isfile(src):
        ##############
        # Parse file #
        ##############
        group = os.path.basename(src).split("_")[0].upper()
        with open(src, "rb") as f:
            lines = f.readlines()
        is_body = False
        x = []
        # 3 significance tests per hypothesis in corresponding order of the
        # file
        # 1-sample tests
        tests1 = ["1-sample Mann-Whitney U", "1-sample Wilcoxon Signed Rank",
                  "1-sample Baum-Welch"]
        # 2-sample tests
        tests2 = ["2-sample Mann-Whitney U", "2-sample Wilcoxon rank sum",
                  "2-sample Baum-Welch"]
        ps = {}
        i = 0
        # Parse file
        for line in lines:
            # Remove \n
            line = line.rstrip()
            # Current hypothesis to which we add p-values; 1-based counting
            hypo = 0
            # Skip header and only read body
            if is_body:
                content = line.split(",")

                # Skip k entry at beginning
                for idx, p in enumerate(content[1:]):
                    # Find out to which hypothesis the current p-value belongs
                    if idx % len(tests1) == 0:
                        hypo += 1
                    # Remove * or ** from significant entries
                    if p.endswith("**"):
                        p = p[:-2]
                    elif p.endswith("*"):
                        p = p[:-1]
                    p = float(p)
                    # Find out the test's name used for the idx-th entry -
                    # choose from 2-sample or 1-sample tests
                    if idx > 8:
                        test = tests2[idx % len(tests1)]
                    else:
                        test = tests1[idx % len(tests1)]
                    ps[hypo][test][group].append(p)
                    i += 1
            # Identify when body begins
            if line.startswith("++++++++++++++++++++++++++++++++"):
                is_body = True
            # Extract from header how many hypotheses exist
            if line.startswith("k 1.H0"):
                # 3 tests per hypotheses, first column contains k
                hypos = (len(line.split(" ")) - 1) / len(tests2)
                print "hypos:", hypos
                # Initialize dict
                for h in xrange(hypos):
                    ps[h+1] = {
                        "1-sample Wilcoxon Signed Rank": {
                                group: []
                        },
                        "1-sample Mann-Whitney U": {
                                group: []
                        },
                        "1-sample Baum-Welch": {
                                group: []
                        },
                        "2-sample Baum-Welch": {
                                group: []
                        },
                        "2-sample Wilcoxon rank sum": {
                                group: []
                        },
                        "2-sample Mann-Whitney U": {
                                group: []
                        }
                    }
        ############
        # Plotting #
        ############
        # 1 plot per group per hypothesis
        for hypo in ps:
            # Inner lists are y-values per significance test - it's name is in
            # <names>
            ys = []
            names = []
            for test in ps[hypo]:
                ys.append(ps[hypo][test][group])
                names.append(test)

            _plot(hypo, group, "SU+MD", ys, names, dst_dir, ending)
    else:
        raise IOError("{} doesn't exist".format(src))


def plot_groups_over_insts_paper(src, dst_dir, ending):
    """
    Reads the p-values from <src> and plots them in <dst_dir>. Has tweaks for
    paper.

    Parameters
    ----------
    src: str - path to txt file storing the p-values.
    dst_dir: str - path to directory where plots should be stored.
    ending: str - file ending for each plot.

    """
    if os.path.isfile(src):
        ##############
        # Parse file #
        ##############
        group = os.path.basename(src).split("_")[0].upper()
        with open(src, "rb") as f:
            lines = f.readlines()
        is_body = False
        x = []
        # 3 significance tests per hypothesis in corresponding order of the
        # file
        # 1-sample tests
        tests1 = ["1-sample Mann-Whitney U", "1-sample Wilcoxon Signed Rank",
                  "1-sample Baum-Welch"]
        # 2-sample tests
        tests2 = ["2-sample Mann-Whitney U", "2-sample Wilcoxon rank sum",
                  "2-sample Baum-Welch"]
        ps = {}
        i = 0
        # Parse file
        for line in lines:
            # Remove \n
            line = line.rstrip()
            # Current hypothesis to which we add p-values; 1-based counting
            hypo = 0
            # Skip header and only read body
            if is_body:
                content = line.split(",")

                # Skip k entry at beginning
                for idx, p in enumerate(content[1:]):
                    # Find out to which hypothesis the current p-value belongs
                    if idx % len(tests1) == 0:
                        hypo += 1
                    # Remove * or ** from significant entries
                    if p.endswith("**"):
                        p = p[:-2]
                    elif p.endswith("*"):
                        p = p[:-1]
                    p = float(p)
                    # Find out the test's name used for the idx-th entry -
                    # choose from 2-sample or 1-sample tests
                    if idx > 8:
                        test = tests2[idx % len(tests1)]
                    else:
                        test = tests1[idx % len(tests1)]
                    ps[hypo][test][group].append(p)
                    i += 1
            # Identify when body begins
            if line.startswith("++++++++++++++++++++++++++++++++"):
                is_body = True
            # Extract from header how many hypotheses exist
            if line.startswith("k 1.H0"):
                # 3 tests per hypotheses, first column contains k
                hypos = (len(line.split(" ")) - 1) / len(tests2)
                print "hypos:", hypos
                # Initialize dict
                for h in xrange(hypos):
                    ps[h+1] = {
                        "1-sample Wilcoxon Signed Rank": {
                                group: []
                        },
                        "1-sample Mann-Whitney U": {
                                group: []
                        },
                        "1-sample Baum-Welch": {
                                group: []
                        },
                        "2-sample Baum-Welch": {
                                group: []
                        },
                        "2-sample Wilcoxon rank sum": {
                                group: []
                        },
                        "2-sample Mann-Whitney U": {
                                group: []
                        }
                    }
        ############
        # Plotting #
        ############
        # 1 plot per group per hypothesis
        for hypo in ps:
            # Inner lists are y-values per significance test - it's name is in
            # <names>
            ys = []
            names = []
            for test in ps[hypo]:
                if test == "2-sample Wilcoxon rank sum":
                    ys.append(ps[hypo][test][group])
                    names.append(test)

            _plot_paper(hypo, group, "SU+MD", ys, names, dst_dir, ending)
    else:
        raise IOError("{} doesn't exist".format(src))


def plot_inter_groups(src, dst_dir, ending):
    """
    Reads the p-values from <src> and plots them in <dst_dir>.

    Parameters
    ----------
    src: str - path to txt file storing the p-values.
    dst_dir: str - path to directory where plots should be stored.
    ending: str - file ending for each plot.

    """
    if os.path.isfile(src):
        ##############
        # Parse file #
        ##############
        with open(src, "rb") as f:
            lines = f.readlines()
        is_body = False
        x = []
        # 3 significance tests per hypothesis in corresponding order of the
        # file
        # 2-sample tests
        tests2 = ["2-sample Mann-Whitney U", "2-sample Wilcoxon rank sum",
                  "2-sample Baum-Welch"]
        ps = {}
        i = 0
        # Parse file
        for line in lines:
            # Remove \n
            line = line.rstrip()
            # Current hypothesis to which we add p-values; 1-based counting
            hypo = 0
            # Skip header and only read body
            if is_body:
                content = line.split(",")

                # Skip k entry at beginning
                for idx, p in enumerate(content[1:]):
                    # Find out to which hypothesis the current p-value belongs
                    if idx % len(tests2) == 0:
                        hypo += 1
                    # Remove * or ** from significant entries
                    if p.endswith("**"):
                        p = p[:-2]
                    elif p.endswith("*"):
                        p = p[:-1]
                    p = float(p)
                    test = tests2[idx % len(tests2)]
                    ps[hypo][test].append(p)
                    i += 1
            # Identify when body begins
            if line.startswith("++++++++++++++++++++++++++++++++"):
                is_body = True
            # Extract from header how many hypotheses exist
            if line.startswith("k 1.H0"):
                # 3 tests per hypotheses, first column contains k
                hypos = (len(line.split(" ")) - 1) / len(tests2)
                print "hypos:", hypos
                # Initialize dict
                for h in xrange(hypos):
                    ps[h+1] = {
                        "2-sample Baum-Welch": [],
                        "2-sample Wilcoxon rank sum": [],
                        "2-sample Mann-Whitney U": []
                    }
        ############
        # Plotting #
        ############
        # 1 plot per group per hypothesis
        for hypo, tests in ps.iteritems():
            # Inner lists are y-values per significance test - it's name is in
            # <names>
            ys = []
            names = []
            for test in tests:
                ys.append(ps[hypo][test])
                names.append(test)
            _plot(hypo, "", "", ys, names, dst_dir, ending, is_inter=True)
    else:
        raise IOError("{} doesn't exist".format(src))


def plot_inter_groups_paper(src, dst_dir, ending):
    """
    Reads the p-values from <src> and plots them in <dst_dir>. Has tweaks for
    paper.

    Parameters
    ----------
    src: str - path to txt file storing the p-values.
    dst_dir: str - path to directory where plots should be stored.
    ending: str - file ending for each plot.

    """
    if os.path.isfile(src):
        ##############
        # Parse file #
        ##############
        with open(src, "rb") as f:
            lines = f.readlines()
        is_body = False
        x = []
        # 3 significance tests per hypothesis in corresponding order of the
        # file
        # 2-sample tests
        tests2 = ["2-sample Mann-Whitney U", "2-sample Wilcoxon rank sum",
                  "2-sample Baum-Welch"]
        ps = {}
        i = 0
        # Parse file
        for line in lines:
            # Remove \n
            line = line.rstrip()
            # Current hypothesis to which we add p-values; 1-based counting
            hypo = 0
            # Skip header and only read body
            if is_body:
                content = line.split(",")

                # Skip k entry at beginning
                for idx, p in enumerate(content[1:]):
                    # Find out to which hypothesis the current p-value belongs
                    if idx % len(tests2) == 0:
                        hypo += 1
                    # Remove * or ** from significant entries
                    if p.endswith("**"):
                        p = p[:-2]
                    elif p.endswith("*"):
                        p = p[:-1]
                    p = float(p)
                    test = tests2[idx % len(tests2)]
                    ps[hypo][test].append(p)
                    i += 1
            # Identify when body begins
            if line.startswith("++++++++++++++++++++++++++++++++"):
                is_body = True
            # Extract from header how many hypotheses exist
            if line.startswith("k 1.H0"):
                # 3 tests per hypotheses, first column contains k
                hypos = (len(line.split(" ")) - 1) / len(tests2)
                print "hypos:", hypos
                # Initialize dict
                for h in xrange(hypos):
                    ps[h+1] = {
                        "2-sample Baum-Welch": [],
                        "2-sample Wilcoxon rank sum": [],
                        "2-sample Mann-Whitney U": []
                    }
        ############
        # Plotting #
        ############
        # 1 plot per group per hypothesis
        for hypo, tests in ps.iteritems():
            # Inner lists are y-values per significance test - it's name is in
            # <names>
            ys = []
            names = []
            for test in tests:
                if test == "2-sample Wilcoxon rank sum":
                    ys.append(ps[hypo][test])
                    names.append(test)
            _plot_paper(hypo, "", "", ys, names, dst_dir, ending, is_inter=True)
    else:
        raise IOError("{} doesn't exist".format(src))


def _plot(hypo, group, institution, ys, tests, dst_dir, ending,
               is_inter=False):
    """
    Plot p-values.

    Parameters
    ----------
    hypo: int - hypothesis.
    group: str - M or S.
    institution: str - MD or SU.
    ys: list of lists of float - each inner list represents p-values of 1
    significance test for given hypothesis, group and institution. Its name is
    the i-th entry in <tests>.
    tests: list of str - each entry is the name of a significance test.
    dst_dir: str - directory where plot is stored.
    ending: str - file ending for each plot.
    is_inter: bool - True if H0 between groups of MD and SU should be plotted.

    """
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    # Create custom artists
    artists = []
    print "hypo", hypo
    colors =["blue", "red", "green"]
    # Color index
    idx = 0
    # Displayed test names
    names = []
    # Plot each significance test's calculated p-values
    for y, name in zip(ys, tests):
        # If for the hypothesis some p-values were calculated with a given test
        if len(y) > 0:
            print name
            s = [5 for i in xrange(len(y))]
            x = [i+1 for i in xrange(len(y))]
            ax.scatter(x, y, s=s, color=colors[idx])
            # Entry for legend
            artist = plt.Line2D((0, 1), (0, 0), color=colors[idx])
            artists.append(artist)
            names.append(name)
            # Next color
            idx += 1
    # Plot significance levels
    plt.axhline(y=0.05, color="black", linestyle="--")
    plt.axhline(y=0.01, color="orange", linestyle="--")
    # Hide the right and top spines (lines)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Set title and increase space to plot
    if is_inter:
        title = "p-values for null hypothesis {}".format(hypo)
    else:
        title = "p-values for null hypothesis {} in {}({})"\
            .format(hypo, institution, group)
    plt.title(title)
    ttl = ax.title
    ttl.set_position([.5, 1.03])
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # Set labels of axes
    ax.set_xlabel("#tweets used from beginning and end for H0")
    ax.set_ylabel("p-value")
    # Add legend outside of plot
    # Get artists and labels for legend and chose which ones to display
    handles, labels = ax.get_legend_handles_labels()
    # display = (0,1,2)
    weak = plt.Line2D((0, 1), (0, 0), color="k", linestyle="--", linewidth=1)
    strong = plt.Line2D((0, 1), (0, 0), color="orange", linestyle="--",
                        linewidth=1)
    # Create legend from custom artist/label lists
    ax.legend([handle for i, handle in enumerate(handles)] + artists + [weak,
              strong], [label for i, label in enumerate(labels)] + names +
              ["$\\alpha=0.05$", "$\\alpha=0.01$"], shadow=True,
              bbox_to_anchor=(1, 1.9), fontsize=FONTSIZE)

    # Limits of axes
    plt.xlim(-1, ax.get_xlim()[1])
    plt.ylim(0, ax.get_ylim()[1])
    if is_inter:
        dst = os.path.join(dst_dir, "learning_effect_groups_in_different_"
                                    "institution_h0_{}{}".format(hypo, ending))
    else:
        dst = os.path.join(dst_dir, "{}_learning_effect_groups_in_same_institu"
                                    "tion_h0_{}_{}{}"
                           .format(institution.lower(), hypo, group, ending))
    plt.savefig(dst, bbox_inches='tight', dpi=600)
    plt.close()


def _plot_paper(hypo, group, institution, ys, tests, dst_dir, ending,
               is_inter=False):
    """
    Plot p-values, especially for the paper.

    Parameters
    ----------
    hypo: int - hypothesis.
    group: str - M or S.
    institution: str - MD or SU.
    ys: list of lists of float - each inner list represents p-values of 1
    significance test for given hypothesis, group and institution. Its name is
    the i-th entry in <tests>.
    tests: list of str - each entry is the name of a significance test.
    dst_dir: str - directory where plot is stored.
    ending: str - file ending for each plot.
    is_inter: bool - True if H0 between groups of MD and SU should be plotted.

    """
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    # Create custom artists
    artists = []
    print "hypo", hypo
    colors =["blue", "red", "green"]
    # Color index
    idx = 0
    # Displayed test names
    names = []
    # Plot each significance test's calculated p-values
    for y, name in zip(ys, tests):
        # If for the hypothesis some p-values were calculated with a given test
        if len(y) > 0:
            print name
            s = [5 for i in xrange(len(y))]
            x = [i+1 for i in xrange(len(y))]
            ax.scatter(x, y, s=s, color=colors[idx])
            # Entry for legend
            artist = plt.Line2D((0, 1), (0, 0), color=colors[idx])
            artists.append(artist)
            names.append(name)
            # Next color
            idx += 1
    # Plot significance levels
    plt.axhline(y=0.05, color="black", linestyle="--")
    plt.axhline(y=0.01, color="orange", linestyle="--")
    # Hide the right and top spines (lines)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Set title and increase space to plot
    # if is_inter:
    #     title = "p-values for null hypothesis {}".format(hypo)
    # else:
    #     title = "p-values for null hypothesis {} in {}({})"\
    #         .format(hypo, institution, group)
    # plt.title(title)
    # ttl = ax.title
    # ttl.set_position([.5, 1.03])
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # Set labels of axes
    ax.set_xlabel("i")
    ax.set_ylabel("p-value")
    # Add legend outside of plot
    # Get artists and labels for legend and chose which ones to display
    handles, labels = ax.get_legend_handles_labels()
    # display = (0,1,2)
    weak = plt.Line2D((0, 1), (0, 0), color="k", linestyle="--", linewidth=1)
    strong = plt.Line2D((0, 1), (0, 0), color="orange", linestyle="--",
                        linewidth=1)
    # Create legend from custom artist/label lists without test names and legend
    # is inside the plot
    # ax.legend([handle for i, handle in enumerate(handles)] + artists + [weak,
    #           strong], [label for i, label in enumerate(labels)] +
    #           ["$\\alpha=0.05$", "$\\alpha=0.01$"], shadow=True,
    #           loc="best")
    # For paper, move the legends somewhere else as they hide data points
    if is_inter and hypo == 1:
        ax.legend([handle for i, handle in enumerate(handles)] + [weak,
              strong], [label for i, label in enumerate(labels)] +
              ["$\\alpha=0.05$", "$\\alpha=0.01$"], shadow=True,
              loc="upper center", bbox_to_anchor=(0.4, 1.0), fontsize=FONTSIZE)
    elif is_inter and hypo == 5:
        ax.legend([handle for i, handle in enumerate(handles)] + [weak,
              strong], [label for i, label in enumerate(labels)] +
              ["$\\alpha=0.05$", "$\\alpha=0.01$"], shadow=True,
              loc="upper center", bbox_to_anchor=(0.4, 0.85), fontsize=FONTSIZE)
    else:
        ax.legend([handle for i, handle in enumerate(handles)] + [weak,
                  strong], [label for i, label in enumerate(labels)] +
                  ["$\\alpha=0.05$", "$\\alpha=0.01$"], shadow=True,
                  loc="best", fontsize=FONTSIZE)

    # Limits of axes
    plt.xlim(-1, ax.get_xlim()[1])
    plt.ylim(0, 0.2)
    if is_inter:
        dst = os.path.join(dst_dir, "learning_effect_groups_in_different_inst"
                                    "itution_h0_{}_paper{}".format(hypo,
                                                                   ending))
    else:
        dst = os.path.join(dst_dir, "{}_learning_effect_groups_in_same_instit"
                                    "ution_h0_{}_{}_paper{}"
                           .format(institution.lower(), hypo, group, ending))
    plt.savefig(dst, bbox_inches='tight', dpi=600)
    plt.close()


def test_group_intra_significance(groups, names, institution, dst_root,
                                  dataset_type, with_l, k=49):
    """
    Tests null hypotheses for each group of an institution separately.

    Parameters
    ----------
    groups: list of np.array  - each array represents a group of size mxn, i.e.
    m annotators where each one labeled n tweets each (if some labeled less
    tweets, the missing columns are filled with np.nan)
    names: list of str - each string represents the name of a group in <groups>
    in the same order.
    institution - str - name of the institution for which the statistical tests
    are performed.
    dst_root: base directory under which all infsci2017_results are stored.
    dataset_type: str - "raw" or "cleaned".
    with_l: str - "with_l" or "without_l".
    k: int - maximum number of tweets that should be used for the analysis for
    firstk and lastk.

    Null hypotheses to be tested:
    1. there is no difference in the firstk median annotation times in S
    among the annotators of the same institution.
    2. there is no difference in the firstk median annotation times in M
    among the annotators of the same institution.
    3. there is no difference in the firstk median annotation times in L
    among the annotators of the same institution.
    4. there is no difference in the lastk median annotation times in S
    among the annotators of the same institution.
    5. there is no difference in the lastk median annotation times in M
    among the annotators of the same institution.
    6. there is no difference in the lastk median annotation times in L
    among the annotators of the same institution.
    7. there is no difference in the firstk + lastk median annotation times in S
    among the annotators of the same institution.
    8. there is no difference in the firstk + lastk median annotation times in M
    among the annotators of the same institution.
    9. there is no difference in the firstk + lastk median annotation times in L
    among the annotators of the same institution.

    """
    # Python to test against the median
    fname = "{}_learning_effect_groups_in_same_institution_{}_{}.txt"\
        .format(institution, with_l, dataset_type)
    with open(dst_root + fname, "wb") as f:
        # Repeat analysis for each k
        for k in xrange(k):
            # k = 0 doesn't make any sense
            k += 1
            f.write("---------------------------\n")
            f.write("k for firstk and lastk: {}\n".format(k))
            f.write("---------------------------\n")
            for group, name in zip(groups, names):
                print "group:", group.shape
                # Reject groups that only have 1 annotator
                if group.shape[0] < 2:
                    continue
                # Keep first k tweets (= first k columns) for each annotator
                # (= row)
                firstk = group[:, 0:k]
                print "firstk:", firstk.shape
                # print firstk
                # Keep last k tweets (= last k columns) for each annotator
                # (= row) that AREN'T np.nan
                # Remove np.nan values
                eq = group[:, ~np.isnan(group).all(0)]
                gr = eq[~np.isnan(eq).all(1)]
                lastk = gr[:, gr.shape[1]-k:]
                print "lastk:", lastk.shape
                combined = np.append(firstk, lastk, axis=0)
                print "combined:", combined.shape
                # Compute medians per annotator (=row) ignoring entries with nan
                # Use set to randomly order the times
                firstk_medians = list(set(np.nanmedian(firstk, axis=1).T))
                # print "set list", firstk_medians
                # print "orig", np.nanmedian(firstk, axis=1).T

                # Median of medians - single number wrapped into an array for
                # significance tests
                firstk_median = [np.nanmedian(np.array(firstk_medians,
                                                       dtype=float))]
                lastk_medians = list(set(np.nanmedian(lastk, axis=1).T))
                lastk_median = [np.nanmedian(np.array(lastk_medians,
                                                      dtype=float))]
                combined_medians = list(set(np.nanmedian(combined, axis=1).T))
                combined_median = [np.nanmedian(np.array(combined_medians,
                                                         dtype=float))]
                # If there are some annotations available either for firstk or
                # lastk or both
                if len(firstk_medians) > 0 or len(lastk_medians) > 0:
                    # ##########################################################
                    # Mann-Whitney test: doesn't assume continuous distribution
                    # (non-parametric)
                    # ##########################################################
                    # 1.H0/2.H0/3.H0
                    u, p = scipy.stats.mannwhitneyu(firstk_median,
                                                    firstk_medians,
                                                    use_continuity=True,
                                                    alternative="two-sided")
                    f.write("Mann-Whitney {} firstk (group intra institution "
                            "{})\n".format(name, institution))
                    f.write("-----------------------------------------------\n")
                    write_significance(f, u, p)
                    # 4.H0/5.H0/6.H0
                    u, p = scipy.stats.mannwhitneyu(lastk_median,
                                                    lastk_medians,
                                                    use_continuity=True,
                                                    alternative="two-sided")
                    f.write("Mann-Whitney {} lastk (group intra institution "
                            "{})\n".format(name, institution))
                    f.write("-----------------------------------------------\n")
                    write_significance(f, u, p)
                    # 7.H0/8.H0/9.H0
                    u, p = scipy.stats.mannwhitneyu(combined_median,
                                                    combined_medians,
                                                    use_continuity=True,
                                                    alternative="two-sided")
                    f.write("Mann-Whitney {} combined (group intra institution "
                            "{})\n".format(name, institution))
                    f.write("-----------------------------------------------\n")
                    write_significance(f, u, p)

                    # ##########################################################
                    # Wilcoxon ranked sum test: assumes continuous distribution
                    # (but discrete should also work) (non-parametric)
                    # ##########################################################
                    # 1.H0/2.H0/3.H0
                    u, p = scipy.stats.ranksums(firstk_median, firstk_medians)
                    f.write("Wilcoxon-ranked-sum {} firstk (group intra "
                            "institution {})\n".format(name, institution))
                    f.write("-----------------------------------------------\n")
                    write_significance(f, u, p)
                    # 4.H0/5.H0/6.H0
                    u, p = scipy.stats.ranksums(lastk_median, lastk_medians)
                    f.write("Wilcoxon-ranked-sum {} lastk (group intra "
                            "institution {})\n".format(name, institution))
                    f.write("-----------------------------------------------\n")
                    write_significance(f, u, p)
                    # 7.H0/8.H0/9.H0
                    u, p = scipy.stats.ranksums(combined_median,
                                                combined_medians)
                    f.write("Wilcoxon-ranked-sum {} combined (group intra "
                            "institution {})\n".format(name, institution))
                    f.write("-----------------------------------------------\n")
                    write_significance(f, u, p)

                    # ##########################################################
                    # Unpaired student t-test: assumes normal distribution and
                    # continuous distribution (parametric); use Baum-Welch
                    # which assumes different variances of the data to be tested
                    # ##########################################################
                    # 1.H0/2.H0/3.H0
                    u, p = scipy.stats.ttest_ind(firstk_median, firstk_medians,
                                                 equal_var=False)
                    f.write("Baum-Welch {} lastk (group intra institution "
                            "{})\n".format(name, institution))
                    f.write("-----------------------------------------------\n")
                    write_significance(f, u, p)
                    # 4.H0/5.H0/6.H0
                    u, p = scipy.stats.ttest_ind(lastk_median, lastk_medians,
                                                 equal_var=False)
                    f.write("Baum-Welch {} lastk (group intra institution "
                            "{})\n".format(name, institution))
                    f.write("-----------------------------------------------\n")
                    write_significance(f, u, p)
                    # 7.H0/8.H0/9.H0
                    # If M is contained in the data (which doesn't have to be
                    # the case)
                    u, p = scipy.stats.ttest_ind(combined_median,
                                                 combined_medians,
                                                 equal_var=False)
                    f.write("Baum-Welch {} combined (group intra "
                            "institution {})\n".format(name, institution))
                    f.write("-----------------------------------------------\n")
                    write_significance(f, u, p)


def get_total_anno_times(tweet_path, anno_path, cleaned=False, with_l=False):
    """
    Computes total annotation time per annotator for all her tweets and
    store those times per annotator group. Keeps n tweets per annotator.

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
    dict.
    {
        "S": [[total times by anno1], [total times by anno2], ...],
        "M": [[...], [...], ...],
        "L": [[...], [...], ...]
    }

    """
    # Maximum number of tweets annotated by a single person -> add dummy values
    # (np.nan) if an annotator labeled less tweets
    n = 500
    # Without L there are only 150 tweets labeled at most
    if not with_l:
        n = 150
    # Group is key, list are annotation times of all annotators of that group
    times = {"S": [],
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
            # Annotation times for hierarchy levels 2 and 3 might be ignored
            t2 = pseudo_db.ZERO
            t3 = pseudo_db.ZERO
            # First level
            rel_label = labels[0]
            t1 = a_times[0]
            # Discard remaining labels if annotator chose "Irrelevant"
            # Consider other sets of labels iff either the cleaned
            # dataset should be created and the label is "relevant" OR
            # the raw dataset should be used.
            if (cleaned and rel_label != "Irrelevant") or not cleaned:
                l2 = labels[1]
                t2 = a_times[1]
                # Annotator labeled the 3rd set of labels as well
                if l2 == pseudo_db.GROUND_TRUTH["Non-factual"]:
                    t3 = a_times[2]
            total = sum([t1, t2, t3])
            anno_times.append(total)
        # Fill up the list with "None" values if less than <n> values are
        # available for an annotator
        anno_times.extend([None] * (n - len(anno_times)))
        if with_l or group != "L":
            # Make sure that each list contains exactly <n> entries
            assert (len(anno_times) == n)
            times[group].append(anno_times)
    return times


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


def compute_significance_unpaired(labels, stat_path):
    """
    Computes if there is a statistically significant difference in the data
    using the
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

        // For the following hypotheses MD doesn't contain data from MD_Later
        2. H0: median confidence times in MD are drawn from the same
        distribution as median confidence times of MD_LATER

    2) between groups of institutions
        1. H0: median confidence times in MD (S) are drawn from the
        same distribution as median confidence times of SU (S)
        2. H0: median confidence times in MD (M) are drawn from the
        same distribution as median confidence times of SU (M)
        3. H0: median confidence times in MD (L) are drawn from the
        same distribution as median confidence times of SU (L)

        // For the following hypotheses MD doesn't contain data from MD_Later
        4. H0: median confidence times in MD (S) are drawn from the same
        distribution as median confidence times of MD_LATER (S)

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
        if len(later_times) > 0:
            # 1.2. H0
            u, p = scipy.stats.mannwhitneyu(md_times, later_times,
                                            use_continuity=True,
                                            alternative="two-sided")
            f.write("Mann-Whitney (institution median conf times LATER vs. "
                    "MD)\n")
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
        if len(later_times) > 0:
            # 2.4. H0
            u, p = scipy.stats.mannwhitneyu(labels["md"]["S"],
                                            labels["later"]["S"],
                                            use_continuity=True,
                                            alternative="two-sided")
            f.write("Mann-Whitney (S) (group median conf times LATER vs. MD)\n")
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
        if len(later_times) > 0:
            # 1.2. H0
            u, p = scipy.stats.ranksums(md_times, later_times)
            f.write("Wilcoxon-ranked-sum (institution median conf times LATER "
                    "vs. MD)\n")
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
        if len(later_times) > 0:
            # 2.4. H0
            u, p = scipy.stats.ranksums(labels["md"]["S"], labels["later"]["S"])
            f.write("Wilcoxon-ranked-sum (S) (group median conf times LATER "
                    "vs. MD)\n")
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
        if len(later_times) > 0:
            # 2.4. H0
            u, p = scipy.stats.ttest_ind(labels["md"]["S"],
                                         labels["later"]["S"], equal_var=False)
            f.write("Baum-Welch (S) (group median conf times LATER "
                    "vs. MD)\n")
            f.write("----------------------------------------------------\n")
            write_significance(f, u, p)


def analyze_results(src_dir, cleaned=True):
    """
    Analyze which hypotheses (H0 1a-c, H0 2a-c, H0 3a-c, see top) yield
    significant infsci2017_results.

    Creates a table containing the minimum k for which a hypothesis yields
    significant infsci2017_results in the following readable format:
    #H  | 1 | 2 | 3
    ----------------
    1   | 3 |...
    2   | - |...
    1, 2, 3 refer to the 3 sets of hypotheses and #H corresponds to hypotheses X
    for the respective set of hypotheses.

    Parameters
    ----------
    src_dir: str - directory in which the p-values are stored.
    cleaned: bool - True if only cleaned data should be used. Otherwise raw data
    is used.

    """
    ending = "cleaned.txt"
    if not cleaned:
        ending = "raw.txt"
    dst_name = "significant_results_summary.txt"
    dst = os.path.join(src_dir, dst_name)
    # We have initially 3 sets of hypotheses
    one = "groups_in_same_institution"
    two = "groups_over_all_institutions"
    three = "learning_effect_groups_in_different_institutions"
    with open(dst, "wb") as f:
        # Write header
        header = "{:^8} | ".format("#H0")
        # Each inner list represents 1 column
        cols = []
        for fname in os.listdir(src_dir):
            # If it contains p-values, isn't the table itself and it's
            # either cleaned or raw
            if not fname.startswith("input_") and fname != dst_name and \
                    fname.endswith(ending):
                fpath = os.path.join(src_dir, fname)
                # Institution or annotator group or nothing
                prefix = fname.split("_")[0]
                if one in fname:
                    s = []
                    m = []
                    # if len(prefix) < 3:
                    # S and M exist in the file
                    header += " {:^2} {:^3} |".format("1", prefix + " S")
                    header += " {:^2} {:^3} |".format("1", prefix + " M")
                    # else:
                    # header += " {:^6} |".format("1")
                    # header += " {:^6} |".format("1")
                    # Parse contents
                    is_body = False
                    is_significant_s = False
                    is_significant_m = False
                    with open(fpath, "rb") as g:
                        lines = g.readlines()
                    for line in lines:
                        line = line.rstrip()
                        # If it's data
                        if is_body:
                            content = line.split(",")
                            group = content[0]
                            k = content[1]
                            if group == "S" and not is_significant_s:
                                # Skip group and k and check p-values
                                for p in content[2:]:
                                    # Significant value exists
                                    if p.endswith("*"):
                                        is_significant_s = True
                                        break
                                s.append(k)
                        # Skip remaining iterations
                        if is_significant_s and is_significant_m:
                            break

                        # Header ends
                        if line.startswith("++++++++"):
                            is_body = True
                if two in fname:
                    # if len(prefix) < 3:
                    header += " {:^2} {:^3} |".format("2", prefix)
                    # else:
                    #     header += " {:^6} |".format("2")
                if three in fname:
                    # if len(prefix) < 3:
                    header += " {:^2} {:^3} |".format("3", prefix)
                    # else:
                    #     header += " {:^6} |".format("3")
        # Discard last |
        f.write(header[:-1] + "\n")
        f.write("-"*len(header))


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    # Maximum number of tweets to skip in the beginning to account for
    # the learning effect
    # 49 because we have 50-500 tweets and we need at least 1 different tweet
    # for firstk and lastk for all groups
    n = 150
    # Directories in which figure/statistics will be stored
    FIG_DIR = os.path.join(base_dir, "results", "figures", "learning_effect")
    STAT_DIR = os.path.join(base_dir, "results", "stats", "learning_effect")
    script_dir = os.path.join(base_dir, "scripts")
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    if not os.path.exists(STAT_DIR):
        os.makedirs(STAT_DIR)

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

    # a) without L
    ###########
    # Raw - SU
    compute_learning_effect_significance(
        script_dir, FIG_DIR, STAT_DIR, md_tweets, md_annos, su_tweets,
        su_annos, n=n, cleaned=False)
    # Cleaned - SU
    compute_learning_effect_significance(
        script_dir, FIG_DIR, STAT_DIR, md_tweets, md_annos, su_tweets,
        su_annos, n=n, cleaned=True)
    analyze_results(STAT_DIR)


