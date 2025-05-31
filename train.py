import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

train_dir = os.path.join('train')
test_dir = os.path.join('test')

img_width, img_height = 224, 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# 3. Model Oluşturma (CNN Modeli)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))  # 8 sınıf için softmax

# 4. Model Derleme
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Model Eğitimi
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=25,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stopping]
)

# Modeli SINIFLANDIRMA.h5 olarak kaydet
model_filename = "model"
model_save_path = f"{model_filename}.h5"  # H5 formatında kayıt
model.save(model_save_path)
print(f"Model başarıyla şu konuma kaydedildi: {model_save_path}")

# 6. Model Değerlendirmesi (Test Verisi)
test_loss, test_accuracy = model.evaluate(test_generator)  # steps parametresini kaldırdık
print(f"Test Accuracy: {test_accuracy:.4f}")

# 7. Tahminler ve Sonuçların Analizi
y_pred = model.predict(test_generator)  # steps parametresini kaldırdık
y_pred_classes = np.argmax(y_pred, axis=1)

# Gerçek etiketleri al (test verisinden)
y_true = test_generator.classes

# Sınıf etiketlerini al
class_names = list(test_generator.class_indices.keys())  # <--- class_names'in tanımı

# Karışıklık Matrisi
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Sınıflandırma Raporu
print(classification_report(y_true, y_pred_classes, target_names=class_names))  # classification_report'u da ekledik.

# 8. Eğitim İstatistiklerini Görselleştirme
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# 9.  Örnek Tahminler (İsteğe Bağlı)
def predict_image(img_path, model, class_names, img_width, img_height):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index]

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}, Confidence: {confidence:.2f}")
    plt.axis('off')
    plt.show()

# Örnek tahmin için bir resim yolu belirtin
if test_dir:
    example_image_path = os.path.join(test_dir, class_names[0], os.listdir(os.path.join(test_dir, class_names[0]))[0])
    predict_image(example_image_path, model, class_names, img_width, img_height)
else:
    print("Test klasörü bulunamadı, örnek tahmin gösterilemez.")