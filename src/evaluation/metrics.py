def calculate_accuracy(y_true, y_pred):
    correct_predictions = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    return correct_predictions / len(y_true)

def calculate_precision(y_true, y_pred, positive_label):
    true_positive = sum((y_t == positive_label) and (y_p == positive_label) for y_t, y_p in zip(y_true, y_pred))
    predicted_positive = sum(y_p == positive_label for y_p in y_pred)
    return true_positive / predicted_positive if predicted_positive > 0 else 0

def calculate_recall(y_true, y_pred, positive_label):
    true_positive = sum((y_t == positive_label) and (y_p == positive_label) for y_t, y_p in zip(y_true, y_pred))
    actual_positive = sum(y_t == positive_label for y_t in y_true)
    return true_positive / actual_positive if actual_positive > 0 else 0

def calculate_f1_score(y_true, y_pred, positive_label):
    precision = calculate_precision(y_true, y_pred, positive_label)
    recall = calculate_recall(y_true, y_pred, positive_label)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def calculate_metrics(y_true, y_pred, positive_label):
    accuracy = calculate_accuracy(y_true, y_pred)
    precision = calculate_precision(y_true, y_pred, positive_label)
    recall = calculate_recall(y_true, y_pred, positive_label)
    f1 = calculate_f1_score(y_true, y_pred, positive_label)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }