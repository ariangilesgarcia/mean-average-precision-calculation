import numpy as np
import data_generators


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


"""
    <area alt="" title="" href="#" shape="rect" coords="120,140,287,270" />  # Verdadero positivo
    <area alt="" title="" href="#" shape="rect" coords="778,134,1039,321" /> # Verdadero positivo
    <area alt="" title="" href="#" shape="rect" coords="334,437,675,722" />  # Falso negativo (no poner)
    <area alt="" title="" href="#" shape="rect" coords="50,500,150,650" />  # Falso positivo
"""

pred = [
	{'class': 'cocoa', 'prob': 1.0, 'x1': 120, 'y1': 140, 'x2': 287, 'y2': 270},
	{'class': 'cocoa', 'prob': 1.0, 'x1': 778, 'y1': 134, 'x2': 1039, 'y2': 321},
	# {'class': 'cocoa', 'prob': 1.0, 'x1': 334, 'y1': 437, 'x2': 675, 'y2': 722},
	{'class': 'cocoa', 'prob': 1.0, 'x1': 50, 'y1': 500, 'x2': 150, 'y2': 650},
]

"""
  	<area alt="" title="" href="#" shape="rect" coords="122,138,283,267" />
    <area alt="" title="" href="#" shape="rect" coords="778,134,1039,321" />
	<area alt="" title="" href="#" shape="rect" coords="334,437,675,722" />
"""

gt = [
	{'class': 'cocoa', 'prob': 1.0, 'x1': 122, 'y1': 138, 'x2': 283, 'y2': 267},
	{'class': 'cocoa', 'prob': 1.0, 'x1': 778, 'y1': 134, 'x2': 1039, 'y2': 321},
	{'class': 'cocoa', 'prob': 1.0, 'x1': 334, 'y1': 437, 'x2': 675, 'y2': 722},
]

x = get_map(pred, gt)
print(x)
