import os
MODELS_DIR = 'models/'

if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

MODEL_TF = MODELS_DIR + 'model'
MODEL_NO_QUANT_TFLITE = MODELS_DIR + 'model_no_quant.tflite'
MODEL_TFLITE = MODELS_DIR + 'model.tflite'
MODEL_TFLITE_MICRO = MODELS_DIR + 'model.cc'

"""
Dependencies
"""
import tensorflow as tf

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import math
import ml_helpers
from subprocess import call

def main(show_plots, use_prox_data):
    """
    Generate Data
    """
    SAMPLES = 1000
    if not use_prox_data:
        seed = 1
        np.random.seed(seed)
        tf.random.set_seed(seed)

        x_values = np.random.uniform(low=0, high=2*math.pi, size=SAMPLES).astype(np.float32)
        y_values = np.sin(x_values).astype(np.float32)
    else:
        x_values = [0]*SAMPLES
        y_values = ml_helpers.get_proximity_data(SAMPLES)

    if show_plots:
        plt.subplot(2, 1, 1)
        plt.plot(x_values, y_values, 'b.')

    """
    Add Noise, skip if using proximity sensor data since it's inherently noisy.
    """
    if not use_prox_data:
        y_values += 0.05 * np.random.randn(*y_values.shape)

    if show_plots:
        plt.subplot(2, 1, 2)
        plt.plot(x_values, y_values, '.r')
        plt.show()

    """
    Split the Data
    """
    TRAIN_SPLIT = int(0.6 * SAMPLES)
    TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)

    x_train, x_test, x_validate = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
    y_train, y_test, y_validate = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

    # Double check that our splits add up correctly
    print(f"train:{x_train.size}, validate:{x_validate.size}, test:{x_test.size}, samples:{SAMPLES}")
    assert (x_train.size + x_validate.size + x_test.size) == SAMPLES

    if show_plots:
        plt.plot(x_train, y_train, 'b.', label="Train")
        plt.plot(x_test, y_test, 'r.', label="Test")
        plt.plot(x_validate, y_validate, 'y.', label="Validate")
        plt.legend()
        plt.show()

    """
    Creating and Training a Model
    """
    model = tf.keras.Sequential()

    # First layer takes a scalar input and feeds it through 16 "neurons".  The
    # neurons decide whether to activate based on teh 'relu' activation function.
    model.add(keras.layers.Dense(16, activation='relu', input_shape=(1,)))

    # The new second and thrid layer will help the network learn more complex representations
    model.add(keras.layers.Dense(16, activation='relu'))

    # Final layer is a single neuron, since we want to output a single value.
    model.add(keras.layers.Dense(1))

    # Compile the model using teh standard 'adam' optimizer and teh mean squared error or 'mse' loss function for regression
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=300, batch_size=64, validation_data=(x_validate, y_validate))

    # Save the model to disk
    model.export(MODEL_TF)

    # Draw a graph of the loss, which is the distance between
    # the predicted and actual values during training and validation.
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(train_loss) + 1)

    # Exclude the first few epochs so the graph is easier to read
    SKIP = 100

    if show_plots:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)

        plt.plot(epochs[SKIP:], train_loss[SKIP:], 'g.', label='Training Loss')
        plt.plot(epochs[SKIP:], val_loss[SKIP:], 'b.', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)

        # Draw a graph of mean absolute error, which is another way of 
        # measuring the amount of error in the prediction.
        train_mae = history.history['mae']
        val_mae = history.history['val_mae']

        plt.plot(epochs[SKIP:], train_mae[SKIP:], 'g.', label='Training MAE')
        plt.plot(epochs[SKIP:], val_mae[SKIP:], 'b.', label='Validation MAE')
        plt.title('Training and validation mean absolute error')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend()

        plt.subplots()

        # Calculate and print the loss on our test dataset
        test_loss, test_mae = model.evaluate(x_test, y_test)

        y_test_pred = model.predict(x_test)

        # Make predictions against the actual values 
        plt.clf()
        plt.title('Comparison of predictions and actual values')
        plt.plot(x_test, y_test, 'b.', label='Actual values')
        plt.plot(x_test, y_test_pred, 'r.', label='TF Predicted')
        plt.legend()
        plt.show()

    """
    Generate a TensorFLow Lite Model
    """
    # Convert the model to the TEnsorFlow Lite formate without quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_TF)
    model_no_quant_tflite = converter.convert()

    # Save the model to the disk
    open(MODEL_NO_QUANT_TFLITE, 'wb').write(model_no_quant_tflite )

    # Convert the model to the TensorFlow Lite format with quantization
    def representativeDataset():
        for i in range(500):
            yield([x_train[i].reshape(1,1)])

    # Set the optimzation flag.
    converter.optimzation = [tf.lite.Optimize.DEFAULT]
    # Enforce integer only quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Provide a representative dataset to ensure we quantize correctly.
    converter.representative_dataset = representativeDataset
    model_tflite = converter.convert()

    # Save the model to disk
    open(MODEL_TFLITE, 'wb').write(model_tflite)

    save_cmd = f"""xxd -i {MODEL_TFLITE} > {MODEL_TFLITE_MICRO}"""
    REPLACE_TEXT = MODEL_TFLITE.replace('/', '_').replace('.', '_')
    filter_cmd = f"""sed -i 's'{REPLACE_TEXT}'/g_models/g' {MODEL_TFLITE_MICRO}"""
    write_cmd = f"""cat {MODEL_TFLITE_MICRO}"""

    call(save_cmd, shell=True)
    call(filter_cmd, shell=True)
    call(write_cmd, shell=True)

if __name__ == '__main__':
    show_plots = False
    use_prox_data = False
    main(show_plots, use_prox_data)