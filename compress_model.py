import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10

model_name = 'resnet_cifar10_depth_30ep'

saved_model_dir = './models/' + model_name + '/model'


(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
Y_train = tf.keras.utils.to_categorical(Y_test, 10)
Y_test = tf.keras.utils.to_categorical(Y_test, 10)

# Rescale
X_train = X_train.astype(np.float32)/255. 
X_test = X_test.astype(np.float32)/255. 



ds_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(1).cache().prefetch(tf.data.experimental.AUTOTUNE)
def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(1000):
    # Model has only one input so each data point has one element.
    yield [input_value]

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)  # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

# Save the model.
with open('./models/' + model_name + '/model.tflite', 'wb') as f:
    f.write(tflite_model)

# python tflite_tools/tflite_tools.py -i models/resnet_cifar10_depth_30ep/model.tflite



# def evaluate(model_file, dataset):
#     # Load the quantised TFLite model
#     interpreter = tf.lite.Interpreter(model_path=model_file)
#     interpreter.allocate_tensors()
#     input_info = interpreter.get_input_details()[0]
#     input_index = input_info["index"]
#     scale, offset = input_info["quantization"]
#     output_index = interpreter.get_output_details()[0]["index"]

#     # Push the dataset through the model and compute accuracy
#     accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
#     test_data = dataset.batch(1).as_numpy_iterator()

#     for x, y_true in test_data:
#         # The model expects a quantised input, spanning the entire range of int8
#         x = (x / scale - offset).astype(np.int8)
#         interpreter.set_tensor(input_index, x)
#         interpreter.invoke()
#         y_pred = interpreter.get_tensor(output_index)
#         accuracy.update_state(y_true, y_pred)
#     return accuracy.result()

# accuracy = evaluate("model.tflite", dataset.testing_dataset())
# print(f"accuracy (quantised model) = {accuracy:.4f}")