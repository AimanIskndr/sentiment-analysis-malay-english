def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=None):
    import matplotlib.pyplot as plt
    import numpy as np

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def generate_confusion_matrix(y_true, y_pred, class_names):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    plot_confusion_matrix(cm, classes=class_names)
    plt.savefig('results/confusion_matrices/confusion_matrix.png')
    plt.show()