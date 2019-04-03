def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    TP, FP, FN, TN = 0, 0, 0, 0
    N = len(prediction)

    for (y_pred, y) in zip(prediction, ground_truth):
        if y_pred and y:
            TP += 1
        if y_pred and not y:
            FP += 1
        if not y_pred and y:
            FN += 1
        if not y_pred and not y:
            TN += 1

    if (TP + FP):
        precision = TP / (TP + FP)
    if TP + FN:
        recall = TP / (TP + FN)
    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    accuracy = (TP + TN) / N

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''

    return sum(y_pred == y for (y_pred, y) in zip(prediction, ground_truth)) / len(prediction)
