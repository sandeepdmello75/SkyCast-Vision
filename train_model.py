import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. Setup Data - Pointing to your 'models' folder
# We divide by 255 here to keep pixel values between 0 and 1
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Make sure 'models' folder exists and has subfolders
if not os.path.exists('models'):
    print("Error: 'models' folder not found! Please check your Desktop folder.")
else:
    train_generator = train_datagen.flow_from_directory(
        'models', 
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    # 2. Build the Neural Network
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(3, activation='softmax') 
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 3. Train
    print("Starting Training...")
    model.fit(train_generator, epochs=10)

    # 4. Save the brain
    model.save('weather_classifier.h5')
    print("Success! 'weather_classifier.h5' has been created.")