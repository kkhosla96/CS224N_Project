import pickle
import sys
import os

PREDICTIONS_FOLDER = "paper_results"
PREDICTIONS_FILE = "predictions.pkl"
COTRAINING_FILE = "cotraining_results.pkl"
CNN_FILE = "cnn/cnn_results.pkl"
FC_FILE = "fc/fc_results.pkl"


def main(experiment_name, cotraining, model=None):
	result_file = None
	if cotraining:
		if model is None:
			result_file = COTRAINING_FILE
		else:
			result_file = CNN_FILE if model=="cnn" else FC_FILE
	else:
		result_file = PREDICTIONS_FILE
	file_path = os.path.join(PREDICTIONS_FOLDER, experiment_name, result_file)
	results = pickle.load(open(file_path, "rb"))

	classes = [t[2] for t in results]
	number_predicted_positive = 0
	number_positive_in_test = 0
	accuracy_count = 0
	precision_recall_count = 0
	for i in range(len(results)):
		predicted_class = results[i][2]
		actual_class = results[i][3]
		if predicted_class == actual_class:
			accuracy_count += 1
		if predicted_class == 1 and actual_class == 1:
			precision_recall_count += 1
		number_predicted_positive += predicted_class
		number_positive_in_test += actual_class

	accuracy = accuracy_count / len(classes)
	precision = precision_recall_count / number_predicted_positive
	recall = precision_recall_count / number_positive_in_test

	print("Number predicted positive: {}".format(number_predicted_positive))
	print("Accuracy: {}".format(accuracy))
	print("Precision: {}".format(precision))
	print("Recall: {}".format(recall))
	print("F1: {}".format(2 * precision * recall / (precision + recall)))

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python view_experiment_results.py <experiment_name> [--cotraining] [model_name]")
	else:
		cotraining = len(sys.argv) >= 3 and sys.argv[2] == "--cotraining"
		model = None
		if cotraining and len(sys.argv) == 4:
			model = sys.argv[3]
		main(sys.argv[1], cotraining, model)
