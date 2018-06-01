import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, auc
import pandas as pd

def do_prc_roc(scores, true_labels, file_name='', directory='', plot=True):
    """ Does the PRC curve

    Args :
            scores (list): list of scores from the decision function
            true_labels (list): list of labels associated to the scores
            file_name (str): name of the PRC curve
            directory (str): directory to save the jpg file
            plot (bool): plots the PRC curve or not
    Returns:
            prc_auc (float): area under the under the PRC curve
    """
    precision, recall, thresholds = precision_recall_curve(true_labels, scores)
    prc_auc = auc(recall, precision)

    fpr, tpr, roc_threshold = roc_curve(true_labels, scores)
    auc_score = auc(fpr, tpr)

    if plot:
        plt.figure()
        plt.subplot(211)
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('recall')
        plt.ylabel('precsion')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('PRC curve: AUC=%0.4f'
                            %(prc_auc))

        plt.subplot(212)
        plt.step(fpr,tpr, color='b', alpha=0.2, where='post')
        plt.fill_between(fpr, tpr, step='post', alpha=0.2, color='b')
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('FPR-TPR curve: AUC=%0.4f'
                            %(auc_score))
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory+'/' + file_name + '_roc_prc.pdf')
        plt.close()

    roc_data = pd.DataFrame({'fpr':fpr, 'tpr':tpr, 'threshold':roc_threshold})
    roc_data.to_csv(directory+'/' + file_name + '_roc.csv')

    thresholds = np.append(thresholds, np.inf)
    print(precision.shape, recall.shape, thresholds.shape)

    prc_data = pd.DataFrame({'precision':precision, 'recall':recall, 'threshold':thresholds})

    prc_data.to_csv(directory + '/' + file_name + '_prc.csv')
    return prc_auc


