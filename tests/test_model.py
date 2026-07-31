import tensorflow as tf

model = tf.keras.models.load_model("models/model.h5")

print("Input Shape :", model.input_shape)
print("Output Shape:", model.output_shape)

model.summary()