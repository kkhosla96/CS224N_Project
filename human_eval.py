import io
import utils
import pickle
import numpy
import random

def count_categories(pred_dict):
	tp_set, tn_set, fp_set, fn_set = set(), set(), set(), set()
	for term, pred in pred_dict.items():
		if pred[0] == 0 and pred[1] == 0:
			tn_set.add(term)
		if pred[0] == 0 and pred[1] == 1:
			fn_set.add(term)
		if pred[0] == 1 and pred[1] == 1:
			tp_set.add(term)
		if pred[0] == 1 and pred[1] == 0:
			fp_set.add(term)
	return tp_set, tn_set, fp_set, fn_set

def build_predictions_dict(file_name):
	pred_dict = {}
	with open(file_name, "rb") as f:
		all_predictions = pickle.load(f)
	for t in all_predictions:
		pred_dict[" ".join(t[0])] = (t[2], t[3])
	return pred_dict


def extract_differing_preds(dict1, dict2):
	all_differing_preds = set()
	all_intersecting_terms = set(dict1.keys()) & set(dict2.keys())
	for term in all_intersecting_terms:
		first_pred = dict1[term][0]
		second_pred = dict2[term][0]
		if first_pred != second_pred:
			all_differing_preds.add(term)
	return all_differing_preds

def run_human_evaluation_experiment(base_dict, model_dict, terms, f):
	baseline_score = 0
	model_score = 0
	for term in terms:
		baseline_pred = base_dict[term][0]
		model_pred = model_dict[term][0]
		#user_answer = utils.yes_or_no("Should the term '%s' belong in the glossary?" % (term))
		f.write("Should the term '%s' belong in the glossary?\n" % (term))
		# if user_answer and baseline_pred:
		# 	baseline_score += 1
		# else:
		# 	model_score += 1
	print("Out of %d total disputed terms, the human agreed with the baseline %f%% of the time and agreed with the model %f%% of the time." %(len(terms), 100*baseline_score/len(terms), 100*model_score/len(terms)))

def filter_predicted_false_positives(model_dict, f):
	num_acceptable = 0
	total_fp = 0
	for term, preds in model_dict.items():
		if preds[0] == 1 and preds[1] == 0: #false positive
			total_fp += 1
			#user_answer = utils.yes_or_no("Should the term '%s' belong in the glossary?" % (term))
			f.write("Should the term '%s' belong in the glossary?\n" % (term))
			user_answer = True
			if user_answer:
				num_acceptable += 1
	print("Out of %d total false positive term, the human thought that %d belonged in the glossary for a rate of %f%%." %(total_fp, num_acceptable, 100* num_acceptable/total_fp))
	return num_acceptable

def filter_positives(preds):
	all_positives = set()
	for term, predictions in preds.items():
		if predictions[0] == 1:
			all_positives.add(term)
	return all_positives

def generate_group_comparison_experiment(baseline_pred, model_pred, golds, f, num_groups, terms_per_group):
	baseline_positives = filter_positives(baseline_pred)
	model_positives= filter_positives(model_pred)
	for i in range(num_groups):
		f.write("Here are three groups of terms, each produced by a different model.\n")
		f.write("Model 1: " + str(random.sample(baseline_positives, terms_per_group)) + "\n")
		f.write("Model 2: " + str(random.sample(model_positives, terms_per_group)) + "\n")
		f.write("Model 3: " + str(random.sample(golds, terms_per_group)) + "\n")
		f.write("Please rank the quality of the glossary terms produced by the three models from lowest to highest using a comma separated sequence of numbers (ex: 3,1,2): \n")
		f.write("\n\n\n")

baseline_predictions_file = "./baseline_results/final_predictions.pkl"
model_predicitons_file = "./experiment_results/supervised_learning_deep_averagebert_dr/predictions.pkl"
gold_file = "./data/gold/openstax_biology/all_golds_preprocessed.pkl"

baseline_pred_dict = build_predictions_dict(baseline_predictions_file)
model_pred_dict = build_predictions_dict(model_predicitons_file)
with open(gold_file, "rb") as f:
	gold_terms = pickle.load(f)

terms_to_examine = extract_differing_preds(baseline_pred_dict, model_pred_dict)

#run_human_evaluation_experiment(baseline_pred_dict, model_pred_dict, terms_to_examine)
tp_set, tn_set, fp_set, fn_set = count_categories(model_pred_dict)
print(len(tp_set), len(tn_set), len(fp_set), len(fn_set))

with open("./human_experiments/experiment_6_human_eval.txt", "w") as f:
	#run_human_evaluation_experiment(baseline_pred_dict, model_pred_dict, terms_to_examine, f)
	#num_acceptable = filter_predicted_false_positives(model_pred_dict, f)
	generate_group_comparison_experiment(baseline_pred_dict, model_pred_dict, gold_terms, f, num_groups=10, terms_per_group=5)
#print("Original preicison was %f%%." % (100 * len(tp_set)/(len(tp_set) + len(fp_set))))
#print("Updated precision after human tagging is %f%%." % (100 * (len(tp_set) + num_acceptable)/(len(tp_set) + len(fp_set))))
