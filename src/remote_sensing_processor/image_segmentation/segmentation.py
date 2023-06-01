import numpy as np
import h5py
import tensorflow as tf

from remote_sensing_processor.image_segmentation.models import unet, deeplabv3, vision_transformer


def segmentation_train(x_train, x_val, y_train, y_val, model, model_file, epochs, batch_size, categorical, x_nodata, y_nodata):
    #opening files and getting data shapes
    if isinstance(x_train, str):
        x_train = h5py.File(x_train)
        input_shape = x_train['data'].shape[1]
        input_dims = x_train['data'].shape[3]
    elif isinstance(x_train, np.ndarray):
        input_shape = x_train.shape[1]
        input_dims = x_train.shape[3]
    if isinstance(x_val, str):
        x_val = h5py.File(x_val)
    if isinstance(y_train, str):
        y_train = h5py.File(y_train)
        num_classes = y_train['data'].shape[3]
    elif isinstance(y_train, np.ndarray):
        num_classes = y_train.shape[3]
    if isinstance(y_val, str):
        y_val = h5py.File(y_val)
    #loading model
    if model == 'unet':
        model = unet(input_shape, input_dims, num_classes)
    elif model == 'deeplabv3':
        model = deeplabv3(input_shape, input_dims, num_classes)
    elif model == 'transformer':
        model = vision_transformer(input_shape, input_dims, num_classes)
    #compiling model
    model.compile(optimizer = optimizers.Adam(learning_rate=0.01),
        loss = 'categorical_crossentropy',
        metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.MeanIoU(name = 'IoU'),
        tf.keras.metrics.AUC(name='AUC')])
    #setting callbacks
    callbacks = []
    if model_file != None:
        checkpoint_filepath = model_file
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        callbacks.append(model_checkpoint_callback)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    callbacks.append(early_stopping)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
        patience=1, min_lr=0.001)
    callbacks.append(reduce_lr)
    #setting up data generators
    train_generator = create_hdf5_generator(x_train, y_train, batch_size, categorical, x_nodata, y_nodata)
    val_generator = create_hdf5_generator(x_val, y_val, batch_size, categorical, x_nodata, y_nodata)
    #training model
    history = model.fit(train_generator, epochs = epochs, validation_data = val_generator, callbacks=callbacks)
    #closing files
    if isinstance(x_train, h5py.File):
        x_train.close()
    if isinstance(x_val, h5py.File):
        x_val.close()
    if isinstance(y_train, h5py.File):
        y_train.close()
    if isinstance(y_val, h5py.File):
        y_val.close()
    return model, history
    
    
def segmentation_test(x_test, y_test, model, batch_size, categorical, x_nodata, y_nodata):
    if isinstance(x_test, str):
        x_train = h5py.File(x_train)
    if isinstance(y_test, str):
        y_train = h5py.File(y_train)
    if isinstance(model, str):
        model = tf.keras.models.load_model(model)
    test_generator = create_hdf5_generator(x_test, y_test, batch_size, categorical, x_nodata, y_nodata)
    results = model.evaluate(test_generator)
    return results
        
        
def create_hdf5_generator(x_in, y_in, batch_size, categorical, x_nodata, y_nodata):
    #getting data shape
    if isinstance(x_in, h5py.File):
        db_size = x_in['data'].shape[0]
    elif isinstance(x_in, np.ndarray):
        db_size = x_in.shape[0]
    #getting nodata and categorical attributes
    if isinstance(x_in, h5py.File) and x_nodata == None:
        x_nodata = x_in.attrs['nodata']
    if isinstance(y_in, h5py.File):
        if y_nodata == None:
            y_nodata = y_in.attrs['nodata']
        categorical = y_in.attrs['categorical']
    #iterating
    while True: # loop through the dataset indefinitely
        for i in np.arange(0, db_size, batch_size):
            #reading batch
            if isinstance(x_in, h5py.File):
                x = x_in['data'][i:i+batch_size].astype('float32')
            elif isinstance(x_in, np.ndarray):
                x = x_in[i:i+batch_size].astype('float32')
            if isinstance(y_in, h5py.File):
                y = y_in['data'][i:i+batch_size].astype('float32')
            elif isinstance(y_in, np.ndarray):
                y = y_in[i:i+batch_size].astype('float32')
            #writing x nodata values to y and vice versa
            if x_nodata != None or y_nodata != None:
                for i in range(len(x)):
                    if categorical == True:
                        if x_nodata != None:
                            x[i] = np.where(np.broadcast_to((y[i][:,:,y_nodata:y_nodata+1]), x[i].shape) == 1, x_nodata, x[i])
                        if y_nodata != None:    
                            y[i][:,:,y_nodata:y_nodata+1] = np.where(p.broadcast_to(x[i][0], y[i].shape) == x_nodata, 1, 0)
                    else:
                        if x_nodata != None:
                            x[i] = np.where(np.broadcast_to(y[i], x[i].shape) == y_nodata, x_nodata, x[i])
                        if x_nodata != None:
                            y[i] = np.where(np.broadcast_to(x[i][0], y[i].shape) == x_nodata, y_nodata, y[i])
            yield x, y
    
