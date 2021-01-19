import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10

model_name = 'resnet_cifar10_depth_es_2'
compress = True


saved_model_dir = './models/' + model_name + '/model'


(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
Y_train = tf.keras.utils.to_categorical(Y_test, 10)
Y_test = tf.keras.utils.to_categorical(Y_test, 10)

# Rescale
X_train = X_train.astype(np.float32)/255. 
X_test = X_test.astype(np.float32)/255. 

def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(1000):
    # Model has only one input so each data point has one element.
    yield [input_value]

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)  # path to the SavedModel directory
if compress:
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_data_gen
  # Ensure that if any ops can't be quantized, the converter throws an error
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  # Set the input and output tensors to uint8 (APIs added in r2.3)
  converter.inference_input_type = tf.uint8
  converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

# Save the model.
with open('./models/' + model_name + '/'+ ("compressed-" if compress else "") + 'model.tflite', 'wb') as f:
    f.write(tflite_model)

# python tflite_tools/tflite_tools.py -i models/resnet_cifar10_depth_30ep/model.tflite

ds_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).cache().prefetch(tf.data.experimental.AUTOTUNE)

def progressBar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    if percent != 100:
      print('Progress: %d/%d [%s%s] %d %%' % (current, total, arrow, spaces, percent), end='\r')
    else:
      print('Progress: %d/%d [%s%s] %d %%' % (current, total, arrow, spaces, percent))

def evaluate(model, dataset):
  # Load the quantised TFLite model
  interpreter = tf.lite.Interpreter(model_content=model)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  # Push the dataset through the model and compute accuracy
  accuracy = tf.keras.metrics.CategoricalAccuracy()
  loss = tf.keras.metrics.CategoricalCrossentropy()
  test_data = dataset.batch(1).as_numpy_iterator()

  count = 0
  for x, y_true in test_data:
    count+=1
    progressBar(count, 10000)
    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
      input_scale, input_zero_point = input_details["quantization"]
      x = x / input_scale + input_zero_point
    x = x.astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], x)
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details["index"])
    accuracy.update_state(y_true, y_pred)
    loss.update_state(y_true, y_pred)
  return {'accuracy': accuracy.result(), 'loss': loss.result()}

result = evaluate(tflite_model, ds_test)
print('./models/' + model_name + '/'+ ("compressed-" if compress else "") + 'model.tflite')
print(f"accuracy = {result['accuracy']:.4f}")
print(f"loss = {result['loss']:.4f}")