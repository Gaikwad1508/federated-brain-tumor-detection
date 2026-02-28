import tensorflow as tf
from tensorflow.keras import layers, models, applications

def get_model(model_name, input_shape=(224, 224, 3), num_classes=4):
    """
    Factory function to return a specific AI model based on the name provided.
    Options: 'vgg16', 'resnet50', 'efficientnet', 'mobilenet', 'custom_cnn'
    """
    model_name = model_name.lower()
    
    if model_name == 'vgg16':
        # --- VGG16: A classic, heavyweight image recognition model ---
        # We load the model but leave off its original final layer (include_top=False)
        base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        
        # We "freeze" the core model so it doesn't relearn basic shapes. This saves laptop GPU memory.
        base_model.trainable = False 
        
        # Shrink the 3D image maps down to a flat 1D list
        x = layers.GlobalAveragePooling2D()(base_model.output)
        
        # Add a custom "thinking" layer to learn our specific tumor features
        x = layers.Dense(128, activation='relu')(x)
        
        # Randomly turn off 50% of the neurons during training so it doesn't just memorize the data
        x = layers.Dropout(0.5)(x) 
        
        # The final decision layer that outputs probabilities for our 4 tumor classes
        output = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=base_model.input, outputs=output)
        
        # Compile with a slow learning rate so it carefully finds the right mathematical weights
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
             
    elif model_name == 'resnet50':
        # --- ResNet50: A deep 50-layer model ---
        base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False 
        
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x) 
        
        output = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=base_model.input, outputs=output)
        
        # ResNet is complex, so we use an even slower learning rate to prevent it from failing
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
             
    elif model_name == 'efficientnet':
        # --- EfficientNetB0 (The True Fine-Tuning Blueprint) ---
        inputs = layers.Input(shape=input_shape)
        
        # 👁️ THE CURE FOR BLINDNESS: 
        # EfficientNet expects raw 0-255 pixels, but our clients send 0.0-1.0.
        # We must multiply by 255 BEFORE the base model, not after!
        rescaled_images = layers.Lambda(lambda x: x * 255.0)(inputs)
        
        # Load the base model
        base_model = applications.EfficientNetB0(weights='imagenet', include_top=False)
        
        # 🔓 Partial Unfreeze: Let the AI physically adapt to MRI textures
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False 
            
        # Pass the corrected, rescaled images INTO the base model
        x = base_model(rescaled_images) 
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x) 
        x = layers.Dropout(0.5)(x)
        
        output = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=inputs, outputs=output)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
    elif model_name == 'mobilenet':
        # --- MobileNetV2: Designed specifically for mobile phones and weak hardware ---
        base_model = applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False 
        
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x) 
        
        output = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=base_model.input, outputs=output)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])     
   
    elif model_name == 'custom_cnn':
        # --- Custom Edge-Net: Built from scratch to prove lightweight architectures work ---
        model = models.Sequential([
            layers.InputLayer(input_shape=input_shape),
            
            # Block 1: Shrink the image immediately to prevent GPU memory crashes
            layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'), 
            layers.BatchNormalization(), # Normalizes the data to keep training stable
            layers.MaxPooling2D((2, 2)), # Shrinks the image size further
            
            # Block 2: Micro Depthwise Convolutions (Very efficient feature extraction)
            layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Block 3: Deep Feature Extraction for complex tumor edges
            layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Final Layer: Flatten the data and make a prediction
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5), # Prevent memorization
            layers.Dense(num_classes, activation='softmax')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
              
    else:
        raise ValueError(f"Model '{model_name}' not supported. Choose vgg16, resnet50, efficientnet, mobilenet, or custom_cnn.")
    
    return model

if __name__ == "__main__":
    # A quick script to test if our factory successfully builds the models without crashing
    print("Testing Model Factory...")
    try:
        model = get_model('efficientnet')
        print(f"✅ Successfully created EfficientNet model with {model.count_params():,} parameters.")
        model = get_model('custom_cnn')
        print(f"✅ Successfully created Custom CNN with {model.count_params():,} parameters.")
    except Exception as e:
        print(f"❌ Error: {e}")