from sklearn.metrics import (
    precision_recall_curve, accuracy_score, f1_score, balanced_accuracy_score,
    precision_score, recall_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_confusion_matrix(y_truth, y_pred, save_fig_path='', classes=[], return_matrix=False):
    '''
    Plot Confusion Matrix.
    # Arguments:
        y_truth : Ground Truth Labels (Series or Iterables).
        y_pred : Predictions (Series or Iterables).
        classes : Class labels (eg. model.classes_). Ideally classes is a dictionary with the format
                  {'class_name_0': 0, 'class_name_1': 1....}. Alternatively it can be a list OR 
                  np.ndarray of class names.
        return_matrix : Return Confusion matrix (boolean default: False).
    # Return:
        Confusion matrix (if return_matrix == True).
    '''

    print('Accuracy :{:.1f}%'.format(balanced_accuracy_score(y_truth,y_pred)*100))

    if any(classes):
        if(isinstance(classes, dict)):
            class_ids = sorted(list(classes.values()))
            class_names = [key for i in class_ids for key, value in classes.items() if i == value]

            if (isinstance(y_truth[0], (int, np.integer))):
                conf_mat = confusion_matrix(y_truth, y_pred, labels=class_ids)
            else:
                conf_mat = confusion_matrix(y_truth, y_pred, labels=class_names)

        elif (isinstance(classes, (list, np.ndarray))):
            class_names = classes
            conf_mat = confusion_matrix(y_truth, y_pred, labels=class_names)

        else:
            raise Exception('ERROR : Invalid type for classes. Please use a dict OR list OR np.ndarray')
    
    else:
        class_names = sorted(list(set(y_truth)))
        conf_mat = confusion_matrix(y_truth, y_pred, labels=class_names)
    
    wh = max(4, len(class_names)//2) # Width-Height adjustment
    fig, ax = plt.subplots(figsize=(wh,wh))
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.yticks(va="center")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    if save_fig_path:
        try:
            plt.savefig(save_fig_path, dpi=300, bbox_inches = "tight")
            print('INFO : Saved Image of Confusion Matrix at ', save_fig_path)
        except:
            print('WARNING : Failed to Save Image of Confusion Matrix at ', save_fig_path)

    plt.show()
    
    if return_matrix:
        return conf_mat, fig

def analyse_epoch(history_dict, filepath=''):
    """
    Create two analysis plots for epoch with training loss and accuracy.
    
    # Arguments
        history_dict: dict, a history dictionary containing loss and accuracy for both training and validation sets.
        filepath: str, filepath to save the plots, default to current directory.

    # Returns
        None
    """
    history_dict.keys()
    
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    epochs = history_dict['epoch']
    fig, ax0 = plt.subplots(1, 1, figsize=(9, 9))
    ax0.plot(epochs, loss_values, 'g', label='Training loss')
    ax0.plot(epochs, val_loss_values, 'b', label='Validation loss')
    ax0.set_title('Training and Validation loss')
    ax0.set_xlabel('Epochs')
    ax0.set_ylabel('Loss')
    ax0.legend()

    plt.show()
    if filepath:
        fig.savefig(filepath, dpi=100)