import tensorflow as tf
from tensorflow.keras import layers, models, applications

def get_model(model_name, input_shape=(224, 224, 3), num_classes=4):
    """
    Factory function to return a compiled model based on the name.
    Options: 'vgg16', 'resnet50', 'efficientnet', 'custom_cnn'
    """
    model_name = model_name.lower()
    
    if model_name == 'vgg16':
        base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False  # Freeze the base for Transfer Learning
        x = layers.Flatten()(base_model.output)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=base_model.input, outputs=output)
        
    elif model_name == 'resnet50':
        base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=base_model.input, outputs=output)
        
    elif model_name == 'efficientnet':
        # EfficientNetB0 is the lightweight champion for Federated Learning
        base_model = applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(128, activation='relu')(x) # Smaller dense layer for efficiency
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=base_model.input, outputs=output)
        
    elif model_name == 'mobilenet':
        base_model = applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False # Keep the core frozen
        
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(128, activation='relu')(x)
        
        # 🛡️ THE FIX: Add 50% Dropout to prevent lazy memorization
        x = layers.Dropout(0.5)(x) 
        
        # Final classification layer
        output = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=base_model.input, outputs=output)
        
        # 🐢 THE FIX: Lower the learning rate for precision
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])     
   
    elif model_name == 'custom_cnn':
        # A simple 4-layer CNN for baseline comparison
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
    
    else:
        raise ValueError(f"Model '{model_name}' not supported. Choose vgg16, resnet50, efficientnet, or custom_cnn.")

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Test the factory
    print("Testing Model Factory...")
    try:
        model = get_model('efficientnet')
        print(f"✅ Successfully created EfficientNet model with {model.count_params()} parameters.")
        model = get_model('custom_cnn')
        print(f"✅ Successfully created Custom CNN with {model.count_params()} parameters.")
    except Exception as e:
        print(f"❌ Error: {e}")