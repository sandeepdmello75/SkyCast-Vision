import tensorflow as tf
model = tf.keras.models.load_model('weather_classifier.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
with open('model.tflite', 'wb') as f:
    f.write(converter.convert())
print("Created model.tflite!")