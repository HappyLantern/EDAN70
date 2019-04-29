def preprocess_minimap(minimap):
    """
    Preprocess minimap feature layers by transforming categorical features into a continuous
    space (one-hot encoding in the channel (feature) dimension followed by 1x1 convolution) 
    and by re-scaling numerical values using the log transform.

    Parameters
    ----------
    minimap : Tensor
        A 4D minimap tensor with shape (batch, channels, y, x)

    Returns
    -------
    Tensor
        A preprocessed 4D minimap tensor with shape (batch, x, y, channels)
    """
    
    assert len(minimap.shape) == 4
    assert minimap.shape[1] == len(features.MINIMAP_FEATURES)
    
    preprocessed_minimap = []
    
    for i, feature in enumerate(features.MINIMAP_FEATURES):
        minimap_feature = minimap[:, i, :, :] # minimap[batch, channels, y, x]; transform in channel dimension
        if feature.type == features.FeatureType.CATEGORICAL:
            one_hot = tf.one_hot(
                minimap_feature,
                depth=feature.scale)
            preprocessed_feature = tf.layers.conv2d(
                inputs=one_hot,
                filters=1,
                kernel_size=[1, 1],
                padding="SAME")
            preprocessed_minimap.append(preprocessed_feature)
        elif feature.type == features.FeatureType.SCALAR:
            preprocessed_feature = tf.log(tf.cast(minimap_feature, tf.float32) + 1.,
                               name="log")
            expanded = tf.expand_dims(preprocessed_feature, -1) # insert dim at the end
            preprocessed_minimap.append(expanded)
            
    preprocessed_minimap = tf.concat(preprocessed_minimap, axis=-1)
    return tf.transpose( # return with shape (batch, x, y, channels)
        preprocessed_minimap,
        perm=[0, 2, 1, 3],
        name="transposed_minimap")

def preprocess_screen(screen):
    """
    Preprocess screen feature layers by transforming categorical features into a continuous
    space (one-hot encoding in the channel (feature) dimension followed by 1x1 convolution) 
    and by re-scaling numerical values using the log transform.

    Parameters
    ----------
    minimap : Tensor
        A 4D screen tensor with shape (batch, channels, y, x)

    Returns
    -------
    Tensor
        A preprocessed 4D screen tensor with shape (batch, x, y, channels)
    """
    assert len(screen.shape) == 4
    assert screen.shape[1] == len(features.SCREEN_FEATURES)
    
    preprocessed_screen = []
    
    for i, feature in enumerate(features.SCREEN_FEATURES):
        screen_feature = screen[:, i, :, :] # screen[batch, channel, y, x]; transform in channel dimension
        if feature.type == features.FeatureType.CATEGORICAL:
            one_hot = tf.one_hot(
                screen_feature,
                depth=feature.scale,
                axis=-1)
            preprocessed_feature = tf.layers.conv2d(
                inputs=one_hot,
                filters=1,
                kernel_size=[1, 1],
                padding="SAME")
            preprocessed_screen.append(preprocessed_feature)
        elif feature.type == features.FeatureType.SCALAR:
            preprocessed_feature = tf.log(tf.cast(screen_feature, tf.float32) + 1.,
                               name="log")
            expanded = tf.expand_dims(preprocessed_feature, -1) # insert dim at the end
            preprocessed_screen.append(expanded)
            
    preprocessed_screen = tf.concat(preprocessed_screen, axis=-1)
    return tf.transpose( # return with shape (batch, x, y, channels)
        preprocessed_screen,
        perm=[0, 2, 1, 3],
        name="transposed_screen")