import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Flatten, Dense, Reshape, Concatenate
from tensorflow.keras.layers import LSTM, TimeDistributed, Cropping1D

import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers

        
def get_model(params, summary=False):
    """ Return the network model
    """
    tf.keras.backend.set_floatx('float64')
    model = None
    
    if params["MODEL"] == "model_META-CONV1D-GAMMA":
        xg_shot_shape = (params["N_SHOTS"],
                         params["WHEEL_TRACE_SHAPE"][0]+16*(params["LENGTH_SHOTS"]-1)*int(params["MERGED_SHOTS_GEOM"]),
                         params["WHEEL_TRACE_SHAPE"][1]-params["REMOVE_CENTRAL_BAND"])
        xe_shot_shape = (params["N_SHOTS"]*params["LENGTH_SHOTS"],1+len(params["SHOTS_EXTRA_INFO"]))
        xg_meta_shape = (params["N_META"],
                         params["WHEEL_TRACE_SHAPE"][0]+16*(params["LENGTH_META"]-1)*int(params["MERGED_META_GEOM"]),
                         params["WHEEL_TRACE_SHAPE"][1]-params["REMOVE_CENTRAL_BAND"])
        
        xg_shot = Input(shape=xg_shot_shape, name="xg_shot")
        xe_shot = Input(shape=xe_shot_shape, name="xe_shot")
        xg_meta = Input(shape=xg_meta_shape, name="xg_meta")
        
        xg_tot = Concatenate(axis=1)([xg_shot,xg_meta])
        xg_tot = TimeDistributed(Conv1D(24,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(32,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(64,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(64,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(128,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(128,3, padding='same', activation='tanh', strides=1))(xg_tot)
        xg_tot = TimeDistributed(AveragePooling1D())(xg_tot)
        xg_tot = TimeDistributed(Flatten())(xg_tot)
        xg_shot_p = Cropping1D(cropping=(0,params["N_META"]))(xg_tot)
        xg_meta_p = Cropping1D(cropping=(params["N_SHOTS"],0))(xg_tot)
        
        xe_shot_ = Reshape((params["N_SHOTS"],params["LENGTH_SHOTS"]*(1+len(params["SHOTS_EXTRA_INFO"]))))(xe_shot)
        x_shot_tot = Concatenate(axis=-1)([xg_shot_p,xe_shot_])
        xg_meta_tot = Concatenate(axis=1)([xg_meta_p for i in range(params["N_SHOTS"])])
        
        h = LSTM(128,return_sequences=True)(x_shot_tot)
        h_tot = Concatenate(axis=-1)([xg_meta_tot,h])
        x = TimeDistributed(Dense(64,activation="relu", name = 'fc1'))(h_tot)
        x = TimeDistributed(Dense(2,activation="linear", name = 'fc2'))(x)
        
        out = []
        for i in range(params["N_SHOTS"]):
            t = Reshape((2,))(Cropping1D(cropping=(i,params["N_SHOTS"]-i-1))(x))
            t = tfp.layers.DistributionLambda(lambda t: tfd.Gamma(concentration=1.0 + 1e-3 + tf.math.softplus(t[..., :1]),
                            rate=1e-3 + tf.math.softplus(t[..., 1:]),
                            allow_nan_stats=False),
                            convert_to_tensor_fn=tfp.distributions.Distribution.mode,
                            name='out_{}'.format(i))(t)
            out.append(t)
        
        model = Model([xg_shot, xe_shot, xg_meta], out)
        
    elif params["MODEL"] == "model_META-CONV1D-LOGNORM":
        xg_shot_shape = (params["N_SHOTS"],
                         params["WHEEL_TRACE_SHAPE"][0]+16*(params["LENGTH_SHOTS"]-1)*int(params["MERGED_SHOTS_GEOM"]),
                         params["WHEEL_TRACE_SHAPE"][1]-params["REMOVE_CENTRAL_BAND"])
        xe_shot_shape = (params["N_SHOTS"]*params["LENGTH_SHOTS"],1+len(params["SHOTS_EXTRA_INFO"]))
        xg_meta_shape = (params["N_META"],
                         params["WHEEL_TRACE_SHAPE"][0]+16*(params["LENGTH_META"]-1)*int(params["MERGED_META_GEOM"]),
                         params["WHEEL_TRACE_SHAPE"][1]-params["REMOVE_CENTRAL_BAND"])
        
        xg_shot = Input(shape=xg_shot_shape, name="xg_shot")
        xe_shot = Input(shape=xe_shot_shape, name="xe_shot")
        xg_meta = Input(shape=xg_meta_shape, name="xg_meta")
        
        xg_tot = Concatenate(axis=1)([xg_shot,xg_meta])
        xg_tot = TimeDistributed(Conv1D(24,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(32,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(64,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(64,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(128,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(128,3, padding='same', activation='tanh', strides=1))(xg_tot)
        xg_tot = TimeDistributed(AveragePooling1D())(xg_tot)
        xg_tot = TimeDistributed(Flatten())(xg_tot)
        xg_shot_p = Cropping1D(cropping=(0,params["N_META"]))(xg_tot)
        xg_meta_p = Cropping1D(cropping=(params["N_SHOTS"],0))(xg_tot)
        
        xe_shot_ = Reshape((params["N_SHOTS"],params["LENGTH_SHOTS"]*(1+len(params["SHOTS_EXTRA_INFO"]))))(xe_shot)
        x_shot_tot = Concatenate(axis=-1)([xg_shot_p,xe_shot_])
        xg_meta_tot = Concatenate(axis=1)([xg_meta_p for i in range(params["N_SHOTS"])])
        
        h = LSTM(128,return_sequences=True)(x_shot_tot)
        h_tot = Concatenate(axis=-1)([xg_meta_tot,h])
        x = TimeDistributed(Dense(64,activation="relu", name = 'fc1'))(h_tot)
        x = TimeDistributed(Dense(2,activation="linear", name = 'fc2'))(x)
        
        sigmin = 1e-3
        sigmax = 1.6
        mumin = -7
        
        out = []
        for i in range(params["N_SHOTS"]):
            t = Reshape((2,))(Cropping1D(cropping=(i,params["N_SHOTS"]-i-1))(x))
            t = tfp.layers.DistributionLambda(lambda t: tfd.LogNormal(loc=tf.keras.activations.elu(t[..., :1],-mumin),
                            scale=sigmin + (sigmax-sigmin)*tf.keras.activations.sigmoid(t[..., 1:])),
                            convert_to_tensor_fn=tfp.distributions.Distribution.mode,
                            name='out_{}'.format(i))(t)
            out.append(t)
        
        model = Model([xg_shot, xe_shot, xg_meta], out)
    
    elif params["MODEL"] == "model_META-CONV1D-GAUSSIAN":
        xg_shot_shape = (params["N_SHOTS"],
                         params["WHEEL_TRACE_SHAPE"][0]+16*(params["LENGTH_SHOTS"]-1)*int(params["MERGED_SHOTS_GEOM"]),
                         params["WHEEL_TRACE_SHAPE"][1]-params["REMOVE_CENTRAL_BAND"])
        xe_shot_shape = (params["N_SHOTS"]*params["LENGTH_SHOTS"],1+len(params["SHOTS_EXTRA_INFO"]))
        xg_meta_shape = (params["N_META"],
                         params["WHEEL_TRACE_SHAPE"][0]+16*(params["LENGTH_META"]-1)*int(params["MERGED_META_GEOM"]),
                         params["WHEEL_TRACE_SHAPE"][1]-params["REMOVE_CENTRAL_BAND"])
        
        xg_shot = Input(shape=xg_shot_shape, name="xg_shot")
        xe_shot = Input(shape=xe_shot_shape, name="xe_shot")
        xg_meta = Input(shape=xg_meta_shape, name="xg_meta")
        
        xg_tot = Concatenate(axis=1)([xg_shot,xg_meta])
        xg_tot = TimeDistributed(Conv1D(24,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(32,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(64,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(64,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(128,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(128,3, padding='same', activation='tanh', strides=1))(xg_tot)
        xg_tot = TimeDistributed(AveragePooling1D())(xg_tot)
        xg_tot = TimeDistributed(Flatten())(xg_tot)
        xg_shot_p = Cropping1D(cropping=(0,params["N_META"]))(xg_tot)
        xg_meta_p = Cropping1D(cropping=(params["N_SHOTS"],0))(xg_tot)
        
        xe_shot_ = Reshape((params["N_SHOTS"],params["LENGTH_SHOTS"]*(1+len(params["SHOTS_EXTRA_INFO"]))))(xe_shot)
        x_shot_tot = Concatenate(axis=-1)([xg_shot_p,xe_shot_])
        xg_meta_tot = Concatenate(axis=1)([xg_meta_p for i in range(params["N_SHOTS"])])
        
        h = LSTM(128,return_sequences=True)(x_shot_tot)
        h_tot = Concatenate(axis=-1)([xg_meta_tot,h])
        x = TimeDistributed(Dense(64,activation="relu", name = 'fc1'))(h_tot)
        x = TimeDistributed(Dense(2,activation="linear", name = 'fc2'))(x)
        
        out = []
        for i in range(params["N_SHOTS"]):
            t = Reshape((2,))(Cropping1D(cropping=(i,params["N_SHOTS"]-i-1))(x))
            t = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],
                            scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:])),
                            convert_to_tensor_fn=tfp.distributions.Distribution.mode,
                            name='out_{}'.format(i))(t)
            out.append(t)
        
        model = Model([xg_shot, xe_shot, xg_meta], out)
    
    elif params["MODEL"] == "model_META-CONV1D-GMM":
        xg_shot_shape = (params["N_SHOTS"],
                         params["WHEEL_TRACE_SHAPE"][0]+16*(params["LENGTH_SHOTS"]-1)*int(params["MERGED_SHOTS_GEOM"]),
                         params["WHEEL_TRACE_SHAPE"][1]-params["REMOVE_CENTRAL_BAND"])
        xe_shot_shape = (params["N_SHOTS"]*params["LENGTH_SHOTS"],1+len(params["SHOTS_EXTRA_INFO"]))
        xg_meta_shape = (params["N_META"],
                         params["WHEEL_TRACE_SHAPE"][0]+16*(params["LENGTH_META"]-1)*int(params["MERGED_META_GEOM"]),
                         params["WHEEL_TRACE_SHAPE"][1]-params["REMOVE_CENTRAL_BAND"])
        
        xg_shot = Input(shape=xg_shot_shape, name="xg_shot")
        xe_shot = Input(shape=xe_shot_shape, name="xe_shot")
        xg_meta = Input(shape=xg_meta_shape, name="xg_meta")
        
        xg_tot = Concatenate(axis=1)([xg_shot,xg_meta])
        xg_tot = TimeDistributed(Conv1D(24,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(32,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(64,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(64,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(128,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(128,3, padding='same', activation='tanh', strides=1))(xg_tot)
        xg_tot = TimeDistributed(AveragePooling1D())(xg_tot)
        xg_tot = TimeDistributed(Flatten())(xg_tot)
        xg_shot_p = Cropping1D(cropping=(0,params["N_META"]))(xg_tot)
        xg_meta_p = Cropping1D(cropping=(params["N_SHOTS"],0))(xg_tot)
        
        xe_shot_ = Reshape((params["N_SHOTS"],params["LENGTH_SHOTS"]*(1+len(params["SHOTS_EXTRA_INFO"]))))(xe_shot)
        x_shot_tot = Concatenate(axis=-1)([xg_shot_p,xe_shot_])
        xg_meta_tot = Concatenate(axis=1)([xg_meta_p for i in range(params["N_SHOTS"])])
        
        h = LSTM(128,return_sequences=True)(x_shot_tot)
        h_tot = Concatenate(axis=-1)([xg_meta_tot,h])
        x = TimeDistributed(Dense(64,activation="relu", name = 'fc1'))(h_tot)
        n_components = params["N_COMPONENTS"]
        event_shape = [1]
        params_size = tfp.layers.MixtureNormal.params_size(n_components, event_shape)
        x = TimeDistributed(Dense(params_size,activation="linear", name = 'fc2'))(x)
        
        out = []
        for i in range(params["N_SHOTS"]):
            t = Reshape((params_size,))(Cropping1D(cropping=(i,params["N_SHOTS"]-i-1))(x))
            t = tfp.layers.MixtureNormal(n_components, event_shape, name="out_{}".format(i))(t)
            out.append(t)
        
        model = Model([xg_shot, xe_shot, xg_meta], out)
    
    elif params["MODEL"] == "model_Meta-Conv1D":
        xg_shot_shape = (params["N_SHOTS"],
                         params["WHEEL_TRACE_SHAPE"][0]+16*(params["LENGTH_SHOTS"]-1)*int(params["MERGED_SHOTS_GEOM"]),
                         params["WHEEL_TRACE_SHAPE"][1]-params["REMOVE_CENTRAL_BAND"])
        xe_shot_shape = (params["N_SHOTS"]*params["LENGTH_SHOTS"],1+len(params["SHOTS_EXTRA_INFO"]))
        xg_meta_shape = (params["N_META"],
                         params["WHEEL_TRACE_SHAPE"][0]+16*(params["LENGTH_META"]-1)*int(params["MERGED_META_GEOM"]),
                         params["WHEEL_TRACE_SHAPE"][1]-params["REMOVE_CENTRAL_BAND"])
        
        xg_shot = Input(shape=xg_shot_shape, name="xg_shot")
        xe_shot = Input(shape=xe_shot_shape, name="xe_shot")
        xg_meta = Input(shape=xg_meta_shape, name="xg_meta")
        
        xg_tot = Concatenate(axis=1)([xg_shot,xg_meta])
        xg_tot = TimeDistributed(Conv1D(24,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(32,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(64,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(64,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(128,3, padding='same', activation='relu', strides=1))(xg_tot)
        xg_tot = TimeDistributed(MaxPooling1D())(xg_tot)
        xg_tot = TimeDistributed(Conv1D(128,3, padding='same', activation='tanh', strides=1))(xg_tot)
        xg_tot = TimeDistributed(AveragePooling1D())(xg_tot)
        xg_tot = TimeDistributed(Flatten())(xg_tot)
        xg_shot_p = Cropping1D(cropping=(0,params["N_META"]))(xg_tot)
        xg_meta_p = Cropping1D(cropping=(params["N_SHOTS"],0))(xg_tot)
        
        xe_shot_ = Reshape((params["N_SHOTS"],params["LENGTH_SHOTS"]*(1+len(params["SHOTS_EXTRA_INFO"]))))(xe_shot)
        x_shot_tot = Concatenate(axis=-1)([xg_shot_p,xe_shot_])
        xg_meta_tot = Concatenate(axis=1)([xg_meta_p for i in range(params["N_SHOTS"])])
        
        h = LSTM(128,return_sequences=True)(x_shot_tot)
        h_tot = Concatenate(axis=-1)([xg_meta_tot,h])
        x = TimeDistributed(Dense(64,activation="relu", name = 'fc1'))(h_tot)
        x = TimeDistributed(Dense(1,activation="linear", name = 'fc2'))(x)
        
        out = []
        for i in range(params["N_SHOTS"]):
            out.append(Reshape((1,),name='out_{}'.format(i))(Cropping1D(cropping=(i,params["N_SHOTS"]-i-1))(x)))
        
        model = Model([xg_shot, xe_shot, xg_meta], out)
        
        
        
    if model is None:
        print("Not valid model name")
        raise ValueError
    elif summary:
        model.summary()
    
    return model