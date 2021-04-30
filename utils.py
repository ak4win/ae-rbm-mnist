import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import json

from rbm import RBM

#====================================================================================================
# 
#====================================================================================================
def dump_models(identifier, models):

    raw_json = []
    for model in models:
        dim_v = model.visible_dim
        dim_h = model.hidden_dim
        v_bias = model.v_bias.numpy().tolist()
        h_bias = model.h_bias.numpy().tolist()
        use_gaussian = model.gaussian_hidden_distribution
        weights = model.W.numpy().tolist()

        raw_json.append({
            "dim_v": dim_v,
            "v_bias": v_bias,
            "dim_h":dim_h,
            "h_bias":h_bias,
            "gaussian":use_gaussian,
            "weights": weights
        })

    with open(f'./models/{identifier}.json', 'w') as f:
        f=json.dump(raw_json, f)
#====================================================================================================
# 
#====================================================================================================
def load_models(identifier):
    with open(f'./models/{identifier}.json', 'r') as f:
        raw_json = json.load(f)
    
    models = []
    for raw_model in raw_json:
        models.append(RBM(
            raw_model['dim_v'],
            raw_model['dim_h'],
            gaussian_hidden_distribution=raw_model['gaussian'],
            pretrained_weights=raw_model['weights'],
            pretrained_v_bias=raw_model['v_bias'],
            pretrained_h_bias=raw_model['h_bias']
        ))
    return models
#====================================================================================================
# 
#====================================================================================================
def min_max_scale_data(data):
    mean = np.mean(data)
    scaling_ratio = np.max(data) - np.min(data)
    scaled_data = (data - mean) / scaling_ratio

    return scaled_data, mean, scaling_ratio

def unscale_min_maxed_data(scaled_data, mean, scaling_ratio):
    return scaled_data * scaling_ratio + mean
#====================================================================================================
# 
#====================================================================================================
def display_output(current_x_train, model, d1, d2, num_imgs=8):
    f, axarr = plt.subplots(2,num_imgs,figsize=(20,8))

    for i in range(num_imgs):
        y = np.random.randint(0, current_x_train.shape[0])
        img_unshaped = current_x_train[y]
        img_unshaped = tf.reshape(img_unshaped, (1, -1))
        img = tf.reshape(img_unshaped, (d1,d2))
        axarr[0,i].imshow(img)
        
        pred = model.predict(img_unshaped)

        img = pred.numpy().reshape(d1,d2)
        axarr[1,i].imshow(img)
    plt.show()
#====================================================================================================
# 
#====================================================================================================
def plot_loss(losses):
    fig, ax = plt.subplots(figsize=(14,6), dpi=80)
    ax.plot(losses, 'b', linewidth=2)
    ax.set_title('Model loss', fontsize=16)
    ax.set_ylabel('Loss (mse)')
    ax.set_xlabel('Epoch')
    plt.show()