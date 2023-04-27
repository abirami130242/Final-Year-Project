import numpy as np
from sklearn.metrics import jaccard_score, accuracy_score, confusion_matrix

# Generate some example data for the predictions and ground truth
predictions = np.random.randint(0, 2, size=(100, 100))
ground_truth = np.random.randint(0, 2, size=(100, 100))

# Calculate Intersection over Union (IoU)
def iou_dice_score(ground_truth, predictions, v):
    inter = np.sum(np.logical_and(predictions == v, ground_truth == v))
    aa = np.sum(ground_truth == v)
    bb = np.sum(predictions == v)
    iou = inter / (aa + bb - inter)
    dice = (2 * inter) / (aa + bb)
    #print(inter, aa, bb)
    return iou, dice

# Calculate Pixel Accuracy
acc = accuracy_score(ground_truth.flatten(), predictions.flatten())
print("Pixel Accuracy: ", acc)

# Calculate F1-Score
def f1score(ground_truth, predictions, v):
    tp = np.sum(np.logical_and(predictions == v, ground_truth == v))
    fp = np.sum(np.logical_and(predictions == v, ground_truth != v))
    fn = np.sum(np.logical_and(predictions != v, ground_truth == v))
    tn = 500 * 500 * 3 - tp - fp - fn
    precision = recall = f1 = 0
    if tp + fp != 0: precision = tp / (tp + fp)
    if tp + fn != 0: recall = tp / (tp + fn)
    if precision != 0 and recall != 0: f1 = 2 * (precision * recall) / (precision + recall)
    return (tp, fp, fn, tn, precision, recall, f1)
