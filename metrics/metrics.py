from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score


def metrics_function(y_true, y_pred):
    """
    This function compute all the following metrics (accuracy, bma and kappa)
    between the y_true (ground truth) and y_pred.
    Parameters
    ----------
    y_true Ground truth labels to score the prediction obtained
    y_pred Labels predicted

    Returns
    -------
    Dictionary containing the 3 different metrics.
    """
    metrics = {"accuracy": accuracy_score(y_true, y_pred), "bma": balanced_accuracy_score(y_true, y_pred),
               "kappa": cohen_kappa_score(y_true, y_pred)}

    return metrics
