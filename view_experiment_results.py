import pickle

predictions_file_stem = "./experiment_results/%s"
file_ender = "supervised_learning_deep/predictions.pkl"

results = pickle.load(open(predictions_file_stem % file_ender, "rb"))

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

print(number_predicted_positive)
print(accuracy)
print(precision)
print(recall)


