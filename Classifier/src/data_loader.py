from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def create_generators(base_dir, target_size=(64,64), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        directory=os.path.join(base_dir, 'train_dir'),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    # Only apply rescaling to validation, no augmentation
    val_gen = test_datagen.flow_from_directory(
        directory=os.path.join(base_dir, 'val_dir'),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    test_gen = test_datagen.flow_from_directory(
        directory=os.path.join(base_dir, 'test_dir'),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_gen, val_gen, test_gen