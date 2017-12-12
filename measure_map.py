import os
import glob
import json
import numpy as np
import data_generators
from sklearn.metrics import average_precision_score


def get_map(pred, gt):
	T = {}
	P = {}

	true_positives = 0

	predicted_count = len(pred)
	gt_count = len(gt)

	for bbox in gt:
		bbox['bbox_matched'] = False

	pred_probs = np.array([s['prob'] for s in pred])
	box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

	# For each predicted box sorted by probabilty
	for box_idx in box_idx_sorted_by_prob:

		# Get bbox predicted data
		pred_box = pred[box_idx]
		pred_class = pred_box['class']
		pred_x1 = pred_box['x1']
		pred_x2 = pred_box['x2']
		pred_y1 = pred_box['y1']
		pred_y2 = pred_box['y2']
		pred_prob = pred_box['prob']

		# If class was not predicted yet, create a new list for that class
		if pred_class not in P:
			P[pred_class] = []
			T[pred_class] = []

		# Add the predicted probability to P
		P[pred_class].append(pred_prob)

		# Set found match to False, meaning the prediction was not found yet on GT
		found_match = False

		# For each bbox in GT
		for gt_box in gt:

			# Get GT bbox data
			gt_class = gt_box['class']
			gt_x1 = gt_box['x1']
			gt_x2 = gt_box['x2']
			gt_y1 = gt_box['y1']
			gt_y2 = gt_box['y2']
			gt_seen = gt_box['bbox_matched']

			# If the class does not match the predicted class, continue
			if gt_class != pred_class:
				continue

			# If the bbox was already matched, continue
			if gt_seen:
				continue

			# Else, check if IoU is greater than 0.5
			iou = data_generators.iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
			if iou >= 0.3:
				# Set found_patch and bbox matched to true
				found_match = True
				gt_box['bbox_matched'] = True

				# Add one to the true positives
				true_positives += 1

				break
			else:
				continue

		T[pred_class].append(int(found_match))

	for gt_box in gt:
		#if not gt_box['bbox_matched'] and not gt_box['difficult']:
		if not gt_box['bbox_matched']:
			if gt_box['class'] not in P:
				P[gt_box['class']] = []
				T[gt_box['class']] = []

			T[gt_box['class']].append(1)
			P[gt_box['class']].append(0)

	#import pdb
	#pdb.set_trace()

	false_positives = predicted_count - true_positives
	false_negatives = gt_count - true_positives

	return T, P, true_positives, false_positives, false_negatives


# Input paths
gt_path = './example_data/gt'
pred_path = './example_data/pred'


# Get JSON files
all_json_files = glob.glob(os.path.join(gt_path, '*.json'))


# Get mAP
T = {}
P = {}

for json_file in all_json_files:
	filename = os.path.basename(json_file)

	gt_file = open(json_file, 'r')
	pred_file = open(os.path.join(pred_path, filename), 'r')

	gt_json = json.load(gt_file)
	pred_json = json.load(pred_file)

	t, p, tp, fp, fn = get_map(pred_json, gt_json)

	print(tp, fp, fn)

	for key in t.keys():
		if key not in T:
			T[key] = []
			P[key] = []
		T[key].extend(t[key])
		P[key].extend(p[key])
	all_aps = []
	for key in T.keys():
		ap = average_precision_score(T[key], P[key])
		print('{} AP: {}'.format(key, ap))
		all_aps.append(ap)
	print('mAP = {}'.format(np.mean(np.array(all_aps))))
