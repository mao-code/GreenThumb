from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import RMSprop

# Normal CNN
def build_model(input_shape=(64,64,3), num_classes=30):
    model = Sequential([
        # Input & 1st Hidden Layer: Convolutional + Max Pooling
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        Dropout(0.25), 
        MaxPool2D((2,2), strides=2),

        # 2nd Hidden Layer: Convolutional + Max Pooling
        Conv2D(64, (3,3), activation='relu', padding='same'),
        Dropout(0.25), 
        MaxPool2D((2,2), strides=2),

        # 3rd Hidden Layer: Convolutional + Max Pooling
        Conv2D(128, (3,3), activation='relu', padding='same'),
        Dropout(0.25), 
        MaxPool2D((2,2), strides=2),

        # Flatten layer
        Flatten(),
        
        # Fully Connected Layer
        Dense(512, activation='relu'),

        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Transfer Learning
def build_transfer_model(image_size=224, num_classes=30, trainable_blocks=None):
    vgg = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(image_size, image_size, 3),
        pooling='max'
    )

    # Freeze all layers
    for layer in vgg.layers:
        layer.trainable = False

    # Set specific number of layers as trainable
    if trainable_blocks:
        for layer in vgg.layers:
            for block in trainable_blocks:
                if layer.name.startswith(block):
                    layer.trainable = True

    model = Sequential([
        vgg,
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=RMSprop(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model