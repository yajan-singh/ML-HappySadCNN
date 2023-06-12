import tensorflow as tf
import downloadData
import parseData

if __name__ == '__main__':
    Queries = ["Happy People", "Sad People", "Angry People",
               "Surprised People", "Disgusted People", "Fearful People"]

    # Limit GPU and CPU Memory Usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Downlaod Images
    # downloadData.download_images(Queries, limit=100)

    # Remove faulty images
    print("\nRemoving Faulty Images\n")
    downloadData.remove_faulty_images()

    # Parse Data
    train_data, val_data, test_data = parseData.split_data(
        parseData.parse_data())

    # Create Model
    print("\nCreating Model\n")
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(6, activation='softmax'))

    # Compile Model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train Model
    print("\nTraining Model\n")
    model.fit(train_data, epochs=10, validation_data=val_data)

    # Evaluate Model
    print("\nEvaluating Model\n")
    print('\n\n[Loss, Accuracy] --> ' + model.evaluate(test_data) + '\n\n')

    # Save Model
    model.save('model.h5')
    print("\nModel Saved\n")
