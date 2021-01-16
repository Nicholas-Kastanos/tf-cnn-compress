import tensorflow as tf

model_name = 'resnet_cifar10_depth_30ep'

saved_model_dir = './models/' + model_name + '/model'

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)  # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model.
with open('./models/' + model_name + '/model.tflite', 'wb') as f:
    f.write(tflite_model)
