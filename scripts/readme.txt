Dataset
It was obtained using our annotation tool (link).

Structure layout
All scripts are stored in the directory scripts/.
Results will be stored at the same level under results/.

required libraries for the scripts to run:
- Python libraries:
matplotlib 1.4.2
unicodecsv 0.14.1
scipy 0.18.1
numpy 1.12.1
statsmodels 0.8.0
editdistance 0.3.1
to install these package versions, you can use pip as follows:
pip install <package_name>==<version_number>, e.g.
pip install numpy==1.12.1
- R must be installed

How to reproduce our Figures (5, 6, 10-20, 22) with the scripts:
- Figure 5 left: - run scripts/label_distribution_significance.py and the figure is found in results/figures/label_distribution/relevant_irrelevant_factual_non-factual_positive_negative_label_distribution_without_l_with_m_per_group_cleaned.pdf
- Figure 5 right: - run scripts/confidence_distribution_significance.py and the figure is found in results/figures/conf_label_distribution/run high_low_conf_label_distribution_without_l_with_m_per_group_cleaned.pdf
- Figure 6 left: - run scripts/irrelevant_labels_and_confidence_vs_rest_over_time.py and the figure is found in results/figures/irrelevant_vs_rest_md_all/annotation_times_median_cleaned.pdf
- Figure 6 right: - run scripts/irrelevant_labels_and_confidence_vs_rest_over_time.py and the figure is found in results/figures/irrelevant_vs_rest_su_all/annotation_times_median_cleaned.pdf
- Figure 10 left: - run scripts/learning_effect_significance.py and the figure is found in results/figures/learning_effect/md_learning_effect_groups_in_same_institution_h0_4_S_paper_without_l_cleaned.pdf
- Figure 10 left: - run scripts/learning_effect_significance.py and the figure is found in results/figures/learning_effect/su_learning_effect_groups_in_same_institution_h0_4_S_paper_without_l_cleaned
- Figure 11 left: - run scripts/learning_effect_significance.py and the figure is found in results/figures/learning_effect/md_learning_effect_groups_in_same_institution_h0_4_M_paper_without_l_cleaned.pdf
- Figure 11 left: - run scripts/learning_effect_significance.py and the figure is found in results/figures/learning_effect/su_learning_effect_groups_in_same_institution_h0_4_M_paper_without_l_cleaned.pdf
- Figure 12 left: - run scripts/learning_effect_acceleration.py and the figure is found in results/figures/acceleration_median_annotation_time/md_50__s__acceleration_fit_polynomial_degree_3_till_16_cleaned.pdf
- Figure 12 right: - run scripts/learning_effect_acceleration.py and the figure is found in results/figures/acceleration_median_annotation_time/md_50__s__acceleration_fit_polynomial_degree_3_till_25_cleaned.pdf
- Figure 13 left: - run scripts/learning_effect_acceleration.py and the figure is found in results/figures/acceleration_median_annotation_time/md_150__m__acceleration_fit_polynomial_degree_3_till_30_cleaned.pdf
- Figure 13 right: - run scripts/learning_effect_acceleration.py and the figure is found in results/figures/acceleration_median_annotation_time/md_150__m__acceleration_fit_polynomial_degree_3_till_41_cleaned.pdf
- Figure 14 left: - run scripts/learning_effect_acceleration.py and the figure is found in results/figures/acceleration_median_annotation_time/su_50__s__acceleration_fit_polynomial_degree_3_till_16_cleaned.pdf
- Figure 14 right: - run scripts/learning_effect_acceleration.py and the figure is found in results/figures/acceleration_median_annotation_time/su_50__s__acceleration_fit_polynomial_degree_3_till_25_cleaned.pdf
- Figure 15 left: - run scripts/learning_effect_acceleration.py and the figure is found in results/figures/acceleration_median_annotation_time/su_150__m__acceleration_fit_polynomial_degree_3_till_30_cleaned.pdf
- Figure 15 right: - run scripts/learning_effect_acceleration.py and the figure is found in results/figures/acceleration_median_annotation_time/su_150__m__acceleration_fit_polynomial_degree_3_till_41_cleaned.pdf
- Figure 16: - run scripts/anno_time_distribution_significance.py and select the pdf in results/figures/median_annotation_times/all_median_anno_times_without_l_with_m_cleaned.pdf
- Figure 17 left: - run scripts/learning_effect_significance.py and the figure is found in results/figures/learning_effect/learning_effect_groups_in_different_institution_h0_1_paper_without_l_cleaned.pdf
- Figure 17 right: - run scripts/learning_effect_significance.py and the figure is found in results/figures/learning_effect/learning_effect_groups_in_different_institution_h0_10_paper_without_l_cleaned.pdf
- Figure 18 left: - run scripts/learning_effect_significance.py and the figure is found in results/figures/learning_effect/learning_effect_groups_in_different_institution_h0_5_paper_without_l_cleaned.pdf
- Figure 18 right: - run scripts/learning_effect_significance.py and the figure is found in results/figures/learning_effect/learning_effect_groups_in_different_institution_h0_14_paper_without_l_cleaned.pdf
- Figure 19 left: - run scripts/learning_effect_significance.py and the figure is found in results/figures/learning_effect/learning_effect_groups_in_different_institution_h0_37_paper_without_l_cleaned.pdf
- Figure 19 right: - run scripts/learning_effect_significance.py and the figure is found in results/figures/learning_effect/learning_effect_groups_in_different_institution_h0_46_paper_without_l_cleaned.pdf
- Figure 20 left: - run scripts/learning_effect_significance.py and the figure is found in results/figures/learning_effect/learning_effect_groups_in_different_institution_h0_41_paper_without_l_cleaned.pdf
- Figure 20 right: - run scripts/learning_effect_significance.py and the figure is found in results/figures/learning_effect/learning_effect_groups_in_different_institution_h0_50_paper_without_l_cleaned.pdf
- Figure 22 left: - run scripts/label_reliability_simulation.py and the figure is found in results/figures/label_reliability/md_edit_cleaned.pdf
- Figure 22 right: - run scripts/label_reliability_simulation.py and the figure is found in results/figures/label_reliability/su_edit_cleaned.pdf

How to reproduce our Tables with the scripts:
- Table 1: derived from the provided datasets for MD and SU.
- Table 2: - run scripts/within_subject_variability_anova.py Look at the entries "mean within-subjects variability" and "mean between-subjects variability" in the text files.
           - to get the values of columns 1 and 2 and 4 and 5, look at the entries for "LEARNING PHASE MATRIX STATS:" and "REST PHASE MATRIX STATS:" respectively in:
             -- for MD: results/stats/within_subject_variability/md_anova_more_cols_without_fatigue_cleaned.txt
             -- for SU: results/stats/within_subject_variability/su_anova_more_cols_without_fatigue_cleaned.txt
           - to get the values in brackets in columns 2 and 5, look at the entries for "REST PHASE MATRIX STATS:" in:
             -- for MD: results/stats/within_subject_variability/md_anova_more_cols_with_fatigue_cleaned.txt
             -- for SU: results/stats/within_subject_variability/su_anova_more_cols_with_fatigue_cleaned.txt
           - to get the values of columns 3 and 6, look at the entries for "FATIGUE PHASE MATRIX STATS:" in:
             -- for MD: results/stats/within_subject_variability/md_anova_more_cols_with_fatigue_cleaned.txt
             -- for SU: results/stats/within_subject_variability/su_anova_more_cols_with_fatigue_cleaned.txt