import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# Định nghĩa các hằng số
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 25

def create_emotion_model():
    """
    Tạo mô hình nhận diện cảm xúc
    """
    img_input = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    x = Conv2D(32, (3, 3), activation='relu')(img_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(7, activation='softmax')(x)  # 7 classes for emotions
    
    model = Model(inputs=img_input, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_combined_model():
    """
    Tạo mô hình kết hợp emotion và PPG
    """
    # Tạo mô hình emotion
    emotion_model = create_emotion_model()
    
    # Tạo input layer cho PPG
    ppg_input = Input(shape=(9,))  # 9 PPG features
    
    # Emotion branch
    x1 = emotion_model.output
    
    # PPG branch
    x2 = Dense(64, activation='relu')(ppg_input)
    x2 = Dropout(0.3)(x2)
    x2 = Dense(32, activation='relu')(x2)
    
    # Combine features
    combined = concatenate([x1, x2])
    
    # Additional layers for stress detection
    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)  # 2 classes: stressed/not stressed
    
    # Create model
    model = Model(inputs=[emotion_model.input, ppg_input], outputs=outputs)
    
    print("Combined model created successfully!")
    return model

def train_combined_model():
    try:
        print("Loading emotion data...")
        # Load emotion data
        X_img = []
        y_emotion = []
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        for label in emotion_labels:
            label_dir = os.path.join('train', label)
            if os.path.exists(label_dir):
                print(f"Processing {label} images...")
                for img_file in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        X_img.append(img)
                        y_emotion.append(emotion_labels.index(label))
        
        X_img = np.array(X_img) / 255.0
        X_img = np.expand_dims(X_img, -1)
        print(f"Loaded {len(X_img)} emotion images")
        
        print("\nLoading PPG data...")
        # Load PPG data
        ppg_features = np.load('wesad_ppg_features.npy')
        ppg_labels = np.load('wesad_labels.npy')
        ppg_labels = to_categorical(ppg_labels, num_classes=2)
        print(f"Loaded PPG features shape: {ppg_features.shape}")
        print(f"Loaded PPG labels shape: {ppg_labels.shape}")
        
        # Ensure same number of samples
        min_samples = min(len(X_img), len(ppg_features))
        X_img = X_img[:min_samples]
        ppg_features = ppg_features[:min_samples]
        y_stress = ppg_labels[:min_samples]
        print(f"\nUsing {min_samples} samples after matching")
        
        # Split data
        X_img_train, X_img_val, X_ppg_train, X_ppg_val, y_train, y_val = train_test_split(
            X_img, ppg_features, y_stress, test_size=0.2, random_state=42
        )
        
        print("\nCreating and compiling model...")
        # Create and compile model
        model = create_combined_model()
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        model.summary()
        
        print("\nStarting training...")
        # Train model
        history = model.fit(
            [X_img_train, X_ppg_train],
            y_train,
            validation_data=([X_img_val, X_ppg_val], y_val),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1
        )
        
        # Save model
        model.save('combined_stress_detection_model.h5')
        print("\nModel saved successfully!")
        
        return model, history
        
    except Exception as e:
        print(f"Error in train_combined_model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def predict_stress(model, face_image, ppg_features):
    """
    Dự đoán mức độ stress từ ảnh khuôn mặt và dữ liệu PPG
    """
    # Tiền xử lý ảnh
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = cv2.resize(face_image, (IMG_SIZE, IMG_SIZE))
    face_image = face_image / 255.0
    face_image = np.expand_dims(face_image, axis=[0, -1])
    
    # Chuẩn bị dữ liệu PPG
    ppg_features = np.expand_dims(ppg_features, axis=0)
    
    # Dự đoán
    prediction = model.predict([face_image, ppg_features])
    stress_level = np.argmax(prediction[0])
    confidence = prediction[0][stress_level]
    
    return stress_level, confidence

if __name__ == "__main__":
    # Train mô hình kết hợp
    model, history = train_combined_model() 