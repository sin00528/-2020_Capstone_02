"""
Train our RNN on extracted features or images.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
import time
import os.path
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import matplotlib.pyplot as plt


def train(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=100):
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(filepath='Data/checkpoints/weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('Data', 'logs', model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=100)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('Data', 'logs', model + '-' + 'training-' + \
        str(timestamp) + '.log'))

    # Get the data and process it.
    if image_shape is None: 
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    if load_to_memory == True:
        # Get data.
        xmin = 1
        xmax = 9
        X, y = data.get_all_sequences_in_memory('train', data_type)
        y = (y-xmin)/(xmax -xmin)
       
        X_val, y_val = data.get_all_sequences_in_memory('val', data_type)
        # import pdb; pdb.set_trace()
        y_val = (y_val-xmin)/(xmax -xmin)
        
        
        
        # 501760
   
    else:
        # Get generators.
        generator = data.frame_generator(batch_size, 'train', data_type)
        val_generator = data.frame_generator(batch_size, 'val', data_type)
    # import pdb; pdb.set_trace()
    # Get the model.
    rm = ResearchModels(data.classes, model, seq_length, saved_model)

    # Fit!
    if load_to_memory == True:
        # Use standard fit.
        history = rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1,
            #callbacks=[tb, early_stopper, csv_logger]
            callbacks=[tb, early_stopper, checkpointer],
            epochs=nb_epoch)
    else:
        # Use fit generator.
        history = rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            # callbacks=[tb, early_stopper, csv_logger, checkpointer],
            callbacks=[early_stopper, checkpointer],
            validation_data=val_generator,
            validation_steps=1,
            workers=4)
    # import pdb; pdb.set_trace()
    pyplot.plot(history.history['mse'])
    
    # plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    # plotter.plot({'Data': history}, metric = "mse")
    # plt.ylim([0, 1])
    # plt.ylabel('MSE ')

    pyplot.show()
def main():
    folder_name ="Data"
    """These are the main training settings. Set each before running
    this file."""
    if os.path.exists(os.path.join(folder_name,'checkpoints')) == False: 
        os.makedirs(os.path.join(folder_name,'checkpoints'))
        
        
    # model can be one of lrcn, mlp
    model = 'mlp'
    saved_model = None  # None or weights file
    class_limit = None  # int, can be 1-101 or None
    seq_length = 20
    load_to_memory = True  # pre-load the sequences into memory
    batch_size = 32
    nb_epoch = 100

    # Chose images or features and image shape based on network.
    if model in ['lrcn']:
        data_type = 'images'
        image_shape = (80, 80, 3)
    elif model in ['mlp']:
        data_type = 'features'
        image_shape = None
    else:
        raise ValueError("Invalid model. See train.py for options.")

    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()

# import tensorflow_docs as tfdocs
# import tensorflow_docs.plots
# import matplotlib.pyplot as plt
# plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
# plotter.plot({'Basic': history}, metric = "mse")
# plt.ylim([0, 10])
# plt.ylabel('MAE [MPG]')
# plt.show()