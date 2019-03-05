import utils


'''
open a Python shell in the same directory as this file, and run from view_results import *
then, you can access these variables.
'''

labeled_file = "./data/output/single_cnn_labeled_data_first_try.pkl"
labels_file = "./data/output/single_cnn_labels_first_try.pkl"
gold_file = "./data/gold/openstax_biology_gold.pkl"

precision, recall = utils.calculate_precision_and_recall(labeled_file, labels_file, gold_file)

labeled, labels = utils.get_labeled_and_labels(labeled_file, labels_file)
golds = utils.get_gold_terms(gold_file)

predicted_positive = set([' '.join(labeled[i]) for i in range(len(labels)) if labels[i]])
predicted_negative = set([' '.join(labeled[i]) for i in range(len(labels)) if not labels[i]])
