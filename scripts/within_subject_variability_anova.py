"""
To measure annotator agreement, or consensus, we can use as proxy annotation
times. Initially, there are high variances/variability and toward the end
they should be lower.
Within-subjects variability is computed as follows: within-subject treatment
and between-subject treatment are computed according to ANOVA computation
but instead of dividing both quantities (to obtain F-statistics which is similar
to p-value, we use them directly.

1 row in ANOVA contains 2 or 3 columns for the following levels (groups):
learning phase, rest, (fatigue), where the latter only exists in M.
Each annotator is a row in the table and we compute for each of the 2/3 columns
this annotator's median annotation time.
See: https://en.wikipedia.org/wiki/Analysis_of_variance
http://www.marsja.se/four-ways-to-conduct-one-way-anovas-using-python/
http://cba.ualr.edu/smartstat/topics/anova/example.pdf

IMPORTANT
---------
To ensure a normal distribution of annotation times (assumption in ANOVA), we
log2-transform them. See the plots in script plot_anno_time_distribution.py

"""
import os
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import pseudo_db


def compute_variability(stat_dir, md_tw_p, md_a_p, su_tw_p, su_a_p,
                        cleaned=False, with_fatigue=True):
    """
    Computes ANOVA statistics, following the code posted here:
    http://www.marsja.se/four-ways-to-conduct-one-way-anovas-using-python/
    This function computes 2/3 more levels by splitting the 2/3 intervals into
    halves. This way, we can compute differences between "learning" and "rest"
    intervals.

    Parameters
    ----------
    stat_dir: str - directory in which the statistics will be stored.
    and the fit of the 2nd polynomial for the remaining data points should start
    cleaned: bool - True if the data should be cleaned, i.e. if tweet is
    "irrelevant", its remaining labels are ignored for computing average
    annotation times.
    md_tw_p: str - path to MD tweets dataset in csv format.
    md_a_p: str - path to MD annotators dataset in csv format.
    su_tw_p: str - path to SU tweets dataset in csv format.
    su_a_p: str - path to SU annotators dataset in csv format.
    with_fatigue: bool - True if for group M a third treatment
    (group/column/interval) exists, namely
    due to fatigue/boredom. Otherwise only two levels are used.

    Raises
    ------
    ValueError: if to_keep is not 3 or more larger than n, which is necessary to
    fit the 2nd polynomial of degree 3.
    """
    agg_type = "raw"
    fat = "without_fatigue"
    if with_fatigue:
        fat = "with_fatigue"
    if cleaned:
        agg_type = "cleaned"
    # Each inner list represents the first <to_keep> annotation times of a
    # single annotator
    # {institution: {group: [[1, 3], [4, 6, 7, 2]]}}
    times = {
        "md": get_total_anno_times(md_tw_p, md_a_p, cleaned=cleaned),
        "su": get_total_anno_times(su_tw_p, su_a_p, cleaned=cleaned)
    }
    # Intervals for fitting: (start_idx, end_idx_1/start_idx, end_idx)
    # Read learn_s = (0, 8, 16): 2 intervals for learn: tweet 0-7, tweet 8-15
    # learn_s = (0, 8, 16)
    # # Rest S: tweet 16-33, tweet 34-50
    # rest_s = (16, 34, 50)
    # learn_m = (0, 18, 35)
    # rest_m = (35, 93, 150)
    learn_s = (0, 10, 20)
    rest_s = (20, 35, 50)
    learn_m = (0, 20, 40)
    rest_m = (40, 95, 150)
    if with_fatigue:
        # rest_m = (35, 53, 70)
        # # Only for M
        # bored = (70, 111, 150)
        rest_m = (40, 60, 80)
        # # Only for M
        bored = (80, 115, 150)
        print "M ends at", bored
    else:
        print "M ends at", rest_m
    # Count how many rows all matrices together have to merge them later
    rows_all = 0
    # Maximum number of columns over all institutions
    cols_all = 0
    # Stores matrices of groups per institution
    # {inst: [matrix of group 1, matrix of group 2, ...], ...}
    mats = {}
    for inst in times:
        mats[inst] = []
        # Count how many rows all matrices per institution have, to merge them
        # later
        rows_inst = 0
        # Maximum number of columns in institution
        cols_inst = 0
        for group in times[inst]:
            # Number of columns in ANOVA; *2 because each level is split into
            # half
            levels = 2 * 2
            if group == "M" and with_fatigue:
                levels = 3 * 2
            annos = len(times[inst][group])
            rows_all += annos
            rows_inst += annos
            cols_all = max(cols_all, levels)
            cols_inst = max(cols_inst, levels)
            mat = np.ones(shape=(annos, levels))
            # Go through all annotation times per annotator
            for idx, anno in enumerate(times[inst][group]):
                y = times[inst][group][anno]
                # Find median time per interval for this annotator
                vals = []
                if group == "S":
                    learn_median1 = np.median(y[learn_s[0]:learn_s[1]])
                    learn_median2 = np.median(y[learn_s[1]:learn_s[2]])
                    rest_median1 = np.median(y[rest_s[0]:rest_s[1]])
                    rest_median2 = np.median(y[rest_s[1]:rest_s[2]])
                    # print rest_median
                    vals.extend([learn_median1, learn_median2, rest_median1,
                                 rest_median2])
                elif group == "M" and with_fatigue:
                    learn_median1 = np.median(y[learn_m[0]:learn_m[1]])
                    learn_median2 = np.median(y[learn_m[1]:learn_m[2]])
                    rest_median1 = np.median(y[rest_m[0]:rest_m[1]])
                    rest_median2 = np.median(y[rest_m[1]:rest_m[2]])
                    fatigue_median1 = np.median(y[bored[0]:bored[1]])
                    fatigue_median2 = np.median(y[bored[1]:bored[2]])
                    vals.extend([learn_median1, learn_median2, rest_median1,
                                 rest_median2, fatigue_median1,
                                 fatigue_median2])
                else:
                    learn_median1 = np.median(y[learn_m[0]:learn_m[1]])
                    learn_median2 = np.median(y[learn_m[1]:learn_m[2]])
                    rest_median1 = np.median(y[rest_m[0]:rest_m[1]])
                    rest_median2 = np.median(y[rest_m[1]:rest_m[2]])
                    vals.extend([learn_median1, learn_median2, rest_median1,
                                 rest_median2])
                mat[idx] = np.array(vals)

            # ANOVA computation per annotator group
            print "{}({})".format(inst, group)
            dst = os.path.join(stat_dir, "{}_{}_anova_more_cols_{}_{}.txt"
                               .format(inst, group.lower(), fat, agg_type))
            anova_more(mat, dst)
            dst = os.path.join(stat_dir, "{}_{}_std_more_cols_{}_{}.txt"
                               .format(inst, group.lower(), fat, agg_type))
            # compute_std(mat, dst)
            dst = os.path.join(stat_dir, "{}_{}_anova_more_cols_{}_{}.png"
                               .format(inst, group.lower(), fat, agg_type))
            # _plot_anova(means, times[inst][group], dst)
            # Store matrix
            mats[inst].append(mat)

        # ANOVA computation for whole institution, i.e. we
        mat = np.empty((rows_inst, cols_inst))
        mat.fill(np.nan)
        # Add each matrix
        next_empty_row = 0
        for m in mats[inst]:
            # print "M"
            # Insert m into mat starting at row next_empty_row in mat.
            # Copy all columns from m (assumption: m.shape[1] <= mat.shape[1])
            mat[next_empty_row:next_empty_row + m.shape[0], 0:m.shape[1]] = m
            # Next matrix can be added below
            next_empty_row += m.shape[0]
        dst = os.path.join(stat_dir, "{}_anova_more_cols_{}_{}.txt"
                           .format(inst, fat, agg_type))
        anova_more(mat, dst)
        dst = os.path.join(stat_dir, "{}_std_more_cols_{}_{}.txt"
                           .format(inst, fat, agg_type))
        # compute_std(mat, dst)

    # ANOVA computation over all institutions
    mat = np.empty((rows_all, cols_all))
    mat.fill(np.nan)
    # Add each matrix
    next_empty_row = 0
    for inst in mats:
        for m in mats[inst]:
            # Insert m into mat starting at row next_empty_row in mat.
            # Copy all columns from m (assumption: m.shape[1] <= mat.shape[1])
            mat[next_empty_row:next_empty_row + m.shape[0], 0:m.shape[1]] = m
            # Next matrix can be added below
            next_empty_row += m.shape[0]
    print "ALL", mat.shape
    dst = os.path.join(stat_dir, "all_anova_more_cols_{}_{}.txt"
                       .format(fat, agg_type))
    anova_more(mat, dst)
    dst = os.path.join(stat_dir, "all_std_more_cols_{}_{}.txt"
                       .format(fat, agg_type))
    # compute_std(mat, dst)


def anova(mat, dst):
    """
    Performs ANOVA analysis.

    Parameters
    ----------
    mat: numpy.array - each row holds median annotation times for different
    intervals. Columns are the intervals.
    dst: str - path where infsci2017_results will be stored.

    """
    # Ignore all columns containing only invalid elements, e.g. NaN
    mat = np.ma.masked_invalid(mat)
    treatment_means = np.ma.mean(mat, axis=0)
    print "treatment means", treatment_means
    grand_mean = np.ma.mean(mat)
    # print "grand mean", grand_mean
    # sst = ((mat-grand_mean)**2).sum()
    sst = np.ma.sum((mat-grand_mean)**2)
    # print "sst", sst
    # Ignore rows where all values are NaN
    rows = mat[~np.all(np.isnan(mat), axis=1)].shape[0]
    # print "rows", rows
    sstr = np.ma.sum((rows*(treatment_means-grand_mean)**2))
    # print "SSTR", sstr
    sse = sst - sstr
    # print "SSE", sse
    # Number of non-nan elements
    n = np.count_nonzero(~np.isnan(mat))
    # print "n", n
    # n = mat.shape[0] * mat.shape[1]
    # Sample variance can be used to express confidence intervals
    mst = sst / (n-1)
    print "sample variance", mst
    # print "MST", mst
    # Ignore columns that only contain NaNs
    # c = number of columns = number of independent variables
    c = mat[:, ~np.all(np.isnan(mat), axis=0)].shape[1]
    # c = mat.shape[1]
    # print "c", c
    # mean between-subjects variability
    mstr = sstr / (c-1)
    print "between-subjects variability", mstr
    # mean within-subjects variability
    mse = sse / (n-c)
    print "within-subjects variability", mse
    f = mstr / mse
    print "F", f
    df_between = c-1
    df_within = n-c
    p = stats.f.sf(f, df_between, df_within)
    print "p", p
    # http://www.theanalysisfactor.com/effect-size/
    # Eta Squared, however, is used specifically in ANOVA models.
    # Each categorical effect in the model has its own Eta Squared, so you
    # get a specific, intuitive measure of the effect of that variable.
    # The drawback for Eta Squared is that it is a biased measure of population
    # variance explained (although it is accurate for the sample). It always
    # overestimates it.
    eta_sqrd = (sstr + sse)/sst
    # print "eta squared", eta_sqrd
    # This bias gets very small as sample size increases, but for small samples
    # an unbiased effect size measure is Omega Squared. Omega Squared has the
    # same basic interpretation, but uses unbiased measures of the variance
    # components. Because it is an unbiased estimate of population variances,
    # Omega Squared is always smaller than Eta Squared.
    om_sqrd = (sstr - (df_between * mse))/(sst + mse)
    # print "omega squared", om_sqrd
    # Store infsci2017_results
    with open(dst, "wb") as fi:
        fi.write("mean within-subjects variability: {:.8f}\n".format(mse))
        fi.write("mean between-subjects variability: {:.8f}\n".format(mstr))
        fi.write("F-statistic: {:.8f}\n".format(f))
        write_significance(fi, p)
        fi.write("biased effect size eta squared: {:.8f}\n".format(eta_sqrd))
        fi.write("unbiased effect size omega squared: {:.8f}\n".format(om_sqrd))
        # root of sample variance is sample deviation
        fi.write("sample variance: {:.8f}\n".format(mst))
        # http://www.jerrydallal.com/lhsp/slrout.htm
        # Correlation coefficient
        r = mstr / sst
        fi.write("Correlation coefficient: {:.8f}\n".format(r))
        fi.write("Explained randomness (in %): {:.8f}\n".format(r*r*100))
        # http://www.stat.yale.edu/Courses/1997-98/101/anovareg.htm
        std_dev = math.sqrt(mse)
        fi.write("standard deviation (+-): {:.8f}\n".format(std_dev))
    return treatment_means


def anova_more(mat, dst):
    """
    Performs ANOVA analysis.

    Parameters
    ----------
    mat: numpy.array - each row holds median annotation times for different
    intervals. Columns are the intervals.
    dst: str - path where infsci2017_results will be stored.

    """
    # Ignore all columns containing only invalid elements, e.g. NaN
    mat = np.ma.masked_invalid(mat)
    print "FULL MATRIX",
    print mat
    print mat.shape
    with open(dst, "wb") as fi:
        # Perform ANOVA for submatrices
        # 1. In learning phase
        mat1 = mat[:, [0, 1]]
        mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst, _ = \
            _compute_anova(mat1)
        fi.write("LEARNING PHASE MATRIX STATS:\n")
        _write_anova(fi, mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst)
        # 2. In rest phase
        mat1 = mat[:, [2, 3]]
        mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst, _ = \
            _compute_anova(mat1)
        fi.write("\nREST PHASE MATRIX STATS:\n")
        _write_anova(fi, mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst)
        # 3. In fatigue phase
        if mat.shape[1] > 4:
            mat1 = mat[:, [4, 5]]
            mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst, _ = \
                _compute_anova(mat1)
            fi.write("\nFATIGUE PHASE MATRIX STATS:\n")
            _write_anova(fi, mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst)
        # 4. Between 1st learning interval and 1st rest interval
        mat1 = mat[:, [0, 2]]
        mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst, _ = \
            _compute_anova(mat1)
        fi.write("\nLEARNING PHASE 1 VS REST PHASE 1 MATRIX STATS:\n")
        _write_anova(fi, mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst)
        # 5. Between 1st learning interval and 2nd rest interval
        mat1 = mat[:, [0, 3]]
        mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst, _ = \
            _compute_anova(mat1)
        fi.write("\nLEARNING PHASE 1 VS REST PHASE 2 MATRIX STATS:\n")
        _write_anova(fi, mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst)
        # 6. Between 2nd learning interval and 1st rest interval
        mat1 = mat[:, [1, 2]]
        mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst, _ = \
            _compute_anova(mat1)
        fi.write("\nLEARNING PHASE 2 VS REST PHASE 1 MATRIX STATS:\n")
        _write_anova(fi, mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst)
        # 7. Between 2nd learning interval and 2nd rest interval
        mat1 = mat[:, [1, 3]]
        mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst, _ = \
            _compute_anova(mat1)
        fi.write("\nLEARNING PHASE 2 VS REST PHASE 2 MATRIX STATS:\n")
        _write_anova(fi, mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst)
        if mat.shape[1] > 4:
            # 8. Between 1st learning interval and 1st fatigue interval
            mat1 = mat[:, [0, 4]]
            mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst, _ = \
                _compute_anova(mat1)
            fi.write("\nLEARNING PHASE 1 VS FATIGUE PHASE 1 MATRIX STATS:\n")
            _write_anova(fi, mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst)
            # 9. Between 1st learning interval and 2nd fatigue interval
            mat1 = mat[:, [0, 5]]
            mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst, _ = \
                _compute_anova(mat1)
            fi.write("\nLEARNING PHASE 1 VS FATIGUE PHASE 2 MATRIX STATS:\n")
            _write_anova(fi, mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst)
            # 10. Between 2nd learning interval and 1st fatigue interval
            mat1 = mat[:, [1, 4]]
            mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst, _ = \
                _compute_anova(mat1)
            fi.write("\nLEARNING PHASE 2 VS FATIGUE PHASE 1 MATRIX STATS:\n")
            _write_anova(fi, mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst)
            # 11. Between 2nd learning interval and 2nd fatigue interval
            mat1 = mat[:, [1, 5]]
            mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst, _ = \
                _compute_anova(mat1)
            fi.write("\nLEARNING PHASE 2 VS FATIGUE PHASE 2 MATRIX STATS:\n")
            _write_anova(fi, mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst)
            # 12. Between 1st rest interval and 1st fatigue interval
            mat1 = mat[:, [2, 4]]
            mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst, _ = \
                _compute_anova(mat1)
            fi.write("\nREST PHASE 1 VS FATIGUE PHASE 1 MATRIX STATS:\n")
            _write_anova(fi, mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst)
            # 13. Between 1st rest interval and 2nd fatigue interval
            mat1 = mat[:, [2, 5]]
            mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst, _ = \
                _compute_anova(mat1)
            fi.write("\nREST PHASE 1 VS FATIGUE PHASE 2 MATRIX STATS:\n")
            _write_anova(fi, mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst)
            # 14. Between 2nd rest interval and 1st fatigue interval
            mat1 = mat[:, [3, 4]]
            mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst, _ = \
                _compute_anova(mat1)
            fi.write("\nREST PHASE 2 VS FATIGUE PHASE 1 MATRIX STATS:\n")
            _write_anova(fi, mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst)
            # 15. Between 2nd rest interval and 2nd fatigue interval
            mat1 = mat[:, [3, 5]]
            mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst, _ = \
                _compute_anova(mat1)
            fi.write("\nREST PHASE 2 VS FATIGUE PHASE 2 MATRIX STATS:\n")
            _write_anova(fi, mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst)
        # 16 Full matrix ANOVA
        mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst, _ = \
            _compute_anova(mat)
        fi.write("\nFULL MATRIX STATS:\n")
        _write_anova(fi, mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst)


def _compute_anova(mat):
    """
    Computes ANOVA statistics for a matrix.

    Parameters
    -----------
    mat: np.array - data used for ANOVA, i.e. columns = levels, rows =
    median annotation times per level of the respective annotator.

    Returns
    -------
    List.
    All values of ANOVA that will be used in the remainder.

    """
    level_means = np.ma.mean(mat, axis=0)
    # print "level means", level_means
    grand_mean = np.ma.mean(mat)
    # print "grand mean", grand_mean
    # sst = ((mat-grand_mean)**2).sum()
    sst = np.ma.sum((mat-grand_mean)**2)
    # print "sst", sst
    # Ignore rows where all values are NaN
    rows = mat[~np.all(np.isnan(mat), axis=1)].shape[0]
    # print "rows", rows
    sstr = np.ma.sum((rows*(level_means-grand_mean)**2))
    # print "SSTR", sstr
    sse = sst - sstr
    # print "SSE", sse
    # Number of non-nan elements
    n = np.count_nonzero(~np.isnan(mat))
    # print "n", n
    # n = mat.shape[0] * mat.shape[1]
    # Sample variance can be used to express confidence intervals
    mst = sst / (n-1)
    # print "sample variance", mst
    # print "MST", mst
    # Ignore columns that only contain NaNs
    # c = number of columns = number of independent variables
    c = mat[:, ~np.all(np.isnan(mat), axis=0)].shape[1]
    # c = mat.shape[1]
    # print "c", c
    # mean between-subjects variability
    mstr = sstr / (c-1)
    # print "between-subjects variability", mstr
    # mean within-subjects variability
    mse = sse / (n-c)
    # print "MSE", sse, n-c
    # print "within-subjects variability", mse
    f = mstr / mse
    # print "F", f
    df_between = c-1
    df_within = n-c
    p = stats.f.sf(f, df_between, df_within)
    # print "p", p
    # http://www.theanalysisfactor.com/effect-size/
    # Eta Squared, however, is used specifically in ANOVA models.
    # Each categorical effect in the model has its own Eta Squared, so you
    # get a specific, intuitive measure of the effect of that variable.
    # The drawback for Eta Squared is that it is a biased measure of population
    # variance explained (although it is accurate for the sample). It always
    # overestimates it.
    eta_sqrd = (sstr + sse)/sst
    # print "eta squared", eta_sqrd
    # This bias gets very small as sample size increases, but for small samples
    # an unbiased effect size measure is Omega Squared. Omega Squared has the
    # same basic interpretation, but uses unbiased measures of the variance
    # components. Because it is an unbiased estimate of population variances,
    # Omega Squared is always smaller than Eta Squared.
    om_sqrd = (sstr - (df_between * mse))/(sst + mse)
    # print "omega squared", om_sqrd
    return [mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst, level_means]


def _write_anova(fi, mse, mstr, f, p, eta_sqrd, om_sqrd, mst, sst):
    fi.write("----------------------------\n")
    fi.write("mean within-subjects variability: {:.8f}\n".format(mse))
    fi.write("mean between-subjects variability: {:.8f}\n".format(mstr))
    fi.write("F-statistic: {:.8f}\n".format(f))
    write_significance(fi, p)
    fi.write("biased effect size eta squared: {:.8f}\n".format(eta_sqrd))
    fi.write("unbiased effect size omega squared: {:.8f}\n".format(om_sqrd))
    # root of sample variance is sample deviation
    fi.write("sample variance: {:.8f}\n".format(mst))
    # http://www.jerrydallal.com/lhsp/slrout.htm
    # Correlation coefficient
    r = mstr / sst
    fi.write("Correlation coefficient: {:.8f}\n".format(r))
    fi.write("Explained randomness (in %): {:.8f}\n".format(r*r*100))
    # http://www.stat.yale.edu/Courses/1997-98/101/anovareg.htm
    print mse
    if mse > 0:
        std_dev = math.sqrt(mse)
        fi.write("standard deviation (+-): {:.8f}\n".format(std_dev))


def compute_std(mat, dst):
    """
    Computes standard deviation for between-subjects variability and
    within-subjects variability.

    Parameters
    ----------
    mat: numpy.array - each row holds median annotation times for different
    intervals. Columns are the intervals.
    dst: str - path where infsci2017_results will be stored.

    """
    # Ignore all columns containing only invalid elements, e.g. NaN
    mat = np.ma.masked_invalid(mat)
    # print mat
    # print mat.shape
    between_stds = np.std(mat, axis=0)
    mean = np.mean(mat)
    within_stds = np.std(mat, axis=1)
    between_std = np.mean(between_stds)
    within_std = np.mean(within_stds)
    # Compute avg of within/between standard deviations
    # print "between:"
    # print between_stds
    # print "within"
    # print within_stds
    print "BETWEEN"
    print "{} +- {}".format(mean, between_std)
    print "WITHIN"
    print "{} +- {}".format(mean, within_std)
    with open(dst, "wb") as fi:
        fi.write("mean: {:.8f}\n".format(mean))
        fi.write("between-subjects standard deviation: {:.8f}\n"
                 .format(between_std))
        fi.write("within-subjects standard deviation: {:.8f}\n"
                 .format(within_std))


def get_lin_ref_coeffs_from_anova():
    """
    ANOVA and linear regression are doing the same thing. To obtain the
    slope and intercept from ANOVA:
    https://stats.stackexchange.com/questions/44838/how-are-the-standard-errors-of-coefficients-calculated-in-a-regression
    https://stats.stackexchange.com/questions/175246/why-is-anova-equivalent-to-linear-regression
    """


def _plot_anova(means, ys, fig_dst):
    """

    """
    intercept = means[0]
    coeff1 = means[1]
    max_x = 0
    # Create plot
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    # Plot all annotation times per annotator
    for idx, anno in enumerate(ys):
        y = ys[anno]
        x = [i+1 for i in xrange(len(y))]
        s = [10 for b in range(len(x))]
        max_x = max(x[-1], max_x)
        if idx == 0:
            ax.scatter(x, y, color="red", s=s, label="Median annotation time")
        else:
            ax.scatter(x, ys[anno], color="red", s=s)
    # Plot the fit
    x = [i+1 for i in xrange(max_x)]
    f = [intercept + i*(coeff1-intercept) for i in x]
    ax.plot(x, f, color="black", linewidth=2, label="slope={:4.2f}"
            .format(coeff1))
    # Hide the right and top spines (lines)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Title - k+1 because we start with k=0
    # title = "Median annotation times for {}"\
    #     .format(institution.upper())
    # # Set title and increase space to plot
    # plt.title(title)
    # ttl = ax.title
    # ttl.set_position([.5, 1.03])
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # Set labels of axes
    ax.set_xlabel("i-th annotated tweet")
    ax.set_ylabel("Median annotation time in s")
    # Add legend outside of plot
    legend = ax.legend(shadow=True, bbox_to_anchor=(1, 1.5))
    # Limits of axes
    plt.xlim(-1, ax.get_xlim()[1])
    plt.ylim(0, ax.get_ylim()[1])
    plt.savefig(fig_dst, bbox_inches='tight', dpi=300)
    plt.close()


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
        "S": { <anno_name1>: [total times by anno1]...,
               <anno_name2>: [total times by anno2]...,
               ...
        },
        "M": see "S"
        "L": see "S"
    }

    """
    # Group is key, list are annotation times of all annotators of that group
    times = {"S": {},
             "M": {},
             "L": {}
             }
    data = pseudo_db.Data(tweet_path, anno_path)
    # For each annotator
    for anno in data.annotators.all_annos():
        username = anno.get_name()
        group = anno.get_group()
        if username not in times[group]:
            times[group][username] = []
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
            # Transform the time to log_2 because we need normally
            # distributed annotation times for ANOVA
            total = math.log(total if total > 0 else 1)
            times[group][username].append(total)
    # Delete group L
    del times["L"]
    return times


def write_significance(f, p):
    """
    Write formatted significance infsci2017_results (statistic and p-value) given
    an open file handle. Also adds ** (or *) to the p-value if the p-value is
    significant on the 0.01 (or 0.05) significance level.

    Parameters
    ----------
    f: file handle - file in which the line should be written.
    computed.
    p: float - p-value.

    """
    if p < 0.01:
        f.write("p: {:.8f}**\n".format(p))
    elif 0.01 <= p < 0.05:
        f.write("p: {:.8f}*\n".format(p))
    else:
        f.write("p: {:.8f}\n".format(p))


def test_anova():
    """
    Implements the example from
    http://cba.ualr.edu/smartstat/topics/anova/example.pdf
    and
    https://www.easycalculation.com/statistics/eta-square-calculator.php
    to see if infsci2017_results are correct. And surprise, surprise, they are
    correct :)

    """
    # Data from URL 1
    # data = [[643, 469, 484], [655, 427, 456], [702, 525, 402]]
    # stat, p = stats.f_oneway([643, 655, 702], [469, 427, 525], [484, 456, 402])
    # Data from URL 2
    data = [[2, 4], [3, 5]]
    stat, p = stats.f_oneway([2, 3], [4, 5])
    # With NaNs -> different result than f_oneway because we don't remove
    # columns/rows containing NaNs
    # data = [[643, 469, 484, np.nan], [655, 427, 456, 555], [702, 525, 402,
    #                                                            np.nan]]
    # stat, p = stats.f_oneway([643, 655, 702], [469, 427, 525], [484, 456, 402])
    mat = np.array(data)
    anova(mat, "anova_test.txt")
    print "f_oneway: stat: {} p: {}".format(stat, p)


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    # Name of the collection in each DB holding annotator data
    ANNO_COLL_NAME = "user"
    # Name of the collection in each DB holding tweet data
    TWEET_COLL_NAME = "tweets"
    # Directory in which stats will be stored
    STAT_DIR = os.path.join(base_dir, "results", "stats",
                            "within_subject_variability")
    if not os.path.exists(STAT_DIR):
        os.makedirs(STAT_DIR)
    # test_anova()

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

    # Have 2 intervals for learning/rest/boredom, i.e. halve existing intervals
    compute_variability(STAT_DIR, md_tweets, md_annos, su_tweets, su_annos,
                        cleaned=False, with_fatigue=False)
    compute_variability(STAT_DIR, md_tweets, md_annos, su_tweets, su_annos,
                        cleaned=False, with_fatigue=True)
    compute_variability(STAT_DIR, md_tweets, md_annos, su_tweets, su_annos,
                        cleaned=True, with_fatigue=False)
    compute_variability(STAT_DIR, md_tweets, md_annos, su_tweets, su_annos,
                        cleaned=True, with_fatigue=True)
