from keras.losses import categorical_crossentropy
import keras.backend as K


# custom loss functions
def cls_loss(y_true, y_pred):
    loss = categorical_crossentropy(y_true, y_pred)

    return loss

def dispel_loss(y_true, y_pred):
    loss = categorical_crossentropy(y_true, y_pred)

    return loss

def pixel_loss(x_real, x_gen):
    height = 128
    width = 128

    l1_distance = 0
    for i in range(height):
        for j in range(width):
            l1_distance += K.abs(x_gen[i][j] - x_real[i][j])

    loss = l1_distance / (height * width)
    
    return loss

def adv_loss(y_true, y_pred):
    # wasserstein_loss
    loss = K.mean(y_true * y_pred)
    return loss
