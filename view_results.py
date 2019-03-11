import utils


'''
open a Python shell in the same directory as this file, and run from view_results import *
then, you can access these variables.
'''

labeled_file = "./experiment_results/various_gs_with_chapters123/data_files/g_3"
labels_file = "./experiment_results/various_gs_with_chapters123/label_files/g_3"
seed_file = "./data/seed_sets/openstax_biology_chapters123_seed.txt"
gold_file = "./data/gold/openstax_biology_chapters123_gold_simple_lemmatized.pkl"

precision, recall = utils.calculate_precision_and_recall(labeled_file, labels_file, seed_file, gold_file)

labeled, labels = utils.get_labeled_and_labels(labeled_file, labels_file)
golds = utils.get_gold_terms(gold_file)

predicted_positive = [' '.join(labeled[i]) for i in range(len(labels)) if labels[i]]
predicted_negative = [' '.join(labeled[i]) for i in range(len(labels)) if not labels[i]]
