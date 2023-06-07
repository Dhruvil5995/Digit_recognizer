import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt

# Load and preprocess data
def load_data(train_data, test_data):
    # Load train and test data
    train_data = pd.read_csv(train_data)
    test_data = pd.read_csv(test_data)
    print('train', train_data.shape)
    print('test', test_data.shape)

    # Extract labels from train data
    train_y = train_data['label'].astype('float32')

    # Extract features from train data
    train_x = train_data.drop(['label'], axis=1).astype('int32')

    # Convert test data to float32
    test_x = test_data.astype('float32')

    # Reshape and normalize train and test data
    train_x = train_x.values.reshape(-1, 28, 28, 1) / 255.0
    test_x = test_x.values.reshape(-1, 28, 28, 1) / 255.0
    print(train_x.shape)

    # Convert labels to one-hot encoded vectors
    train_y = tf.keras.utils.to_categorical(train_y, 10)
    print(train_y.shape)

    return train_x, train_y, test_x

# Visualize sample images from the training dataset
def visualization(train_x, train_y):
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(10):
        axes[i].imshow(train_x[i].reshape(28, 28), cmap='gray')
        axes[i].set_title('Label: {}'.format(np.argmax(train_y[i])))
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Build and train the model
def model_build(train_x, train_y, batch_size, epochs):
    model = Sequential()

    # Add convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Flatten the output from convolutional layers
    model.add(Flatten())

    # Add fully connected layers
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.50))

    # Output layer
    model.add(Dense(10, activation='softmax'))

    # Print model summary
    model.summary()

    # Compile and train the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)

    return model

#Run code

train_x, train_y, test_x= load_data('train.csv', 'test.csv')
vis=visualization(train_x, train_y)
train = model_build(train_x,train_y,50,25)

# Generate predictions for the test data
predictions = train.predict(test_x)
predicted_labels = np.argmax(predictions, axis=1)

# Print predicted labels
print("Predicted Labels:")
print(predicted_labels)
