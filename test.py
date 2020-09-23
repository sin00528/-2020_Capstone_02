

import numpy as np
import csv
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
from keras.optimizers import Adam
from keras.models import load_model
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, median_absolute_error, mean_squared_log_error, mean_absolute_error
from matplotlib import pyplot
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def main():
    seq_length = 20
    class_limit = None
    image_shape = (80, 80, 3)
    ymin = 1
    ymax = 9
    data = DataSet(
        seq_length=seq_length,
        class_limit=class_limit,
        image_shape=image_shape
    )
    batch_size = 20
    concat = False
    
    # data_type = 'images'
    data_type = 'features'
    
    X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
    y_test1 = np.argmax(y_test, axis=1)


    model = load_model('./Data/checkpoints/weights.hdf5')
    
    optimizer = Adam(lr=1e-3)  # aggressively small learning rate
    crits = ['mse']
    model.compile(loss='mse', optimizer=optimizer, metrics=crits)
    
    # loss, mae, mse = model.evaluate(y_test, X_test, verbose=2)
    test_predictions = model.predict(X_test)
    
    # import pdb; pdb.set_trace()
    a = plt.axes(aspect='equal')
    test_predictions = np.round((test_predictions*(ymax-ymin) + ymin), 1)
    
    mse_valuece = mean_squared_error(y_test[:,0],test_predictions[:,0])
    print("mse_valuece: ", mse_valuece)
    mse_arousal = mean_squared_error(y_test[:,1],test_predictions[:,1])
    print("mse_arousal: ", mse_arousal)
    mse_stress = mean_squared_error(y_test[:,2],test_predictions[:,2])
    print("mse_stress: ", mse_stress)

    MSE = (mse_valuece+mse_arousal+mse_stress*2)/4
    print("MSE", MSE)
  
if __name__ == '__main__':
    main()
