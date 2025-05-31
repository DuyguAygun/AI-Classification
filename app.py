import os
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

MODEL_PATH = 'model.h5'
IMG_WIDTH, IMG_HEIGHT = 224, 224

CLASS_NAMES = ['ADONIS', 'AFRICAN GIANT SWALLOWTAIL', 'AMERICAN SNOOT', 'AN 88',
               'APPOLLO', 'ARCIGERA FLOWER MOTH', 'ATALA', 'ATLAS MOTH']

model = load_model(MODEL_PATH)


def preprocess_image(img):

    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array


def predict_butterfly(img):

    if img is None:
        return None, "Lütfen bir görüntü yükleyin", {}

    processed_img = preprocess_image(img)

    prediction = model.predict(processed_img)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = float(prediction[0][predicted_class_index])

    confidence_scores = {CLASS_NAMES[i]: float(prediction[0][i]) for i in range(len(CLASS_NAMES))}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(CLASS_NAMES, prediction[0])
    ax.set_ylabel('Olasılık')
    ax.set_title(f'Tahmin: {predicted_class} (Güven: {confidence:.2f})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig, f"Tahmin Edilen Tür: {predicted_class} (Güven: {confidence:.2f})", confidence_scores


with gr.Blocks(title="Kelebek Türü Sınıflandırıcı") as demo:
    gr.Markdown("# Kelebek Türü Sınıflandırıcı")
    gr.Markdown("Bu uygulama, bir kelebek görüntüsünü sınıflandırarak türünü tahmin eder.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Kelebek Görüntüsü", type="pil")
            predict_button = gr.Button("Tahmin Et", variant="primary")

        with gr.Column():
            output_plot = gr.Plot(label="Sınıflandırma Sonucu")
            output_label = gr.Textbox(label="Tahmin")
            output_confidence = gr.Json(label="Sınıf Olasılıkları")

    predict_button.click(
        fn=predict_butterfly,
        inputs=[input_image],
        outputs=[output_plot, output_label, output_confidence]
    )

    gr.Examples(
        examples=[
            # ["örnek_görüntü_1.jpg"],
            # ["örnek_görüntü_2.jpg"],
        ],
        inputs=input_image,
    )

# Ana çalıştırma fonksiyonu
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"HATA: Model dosyası ({MODEL_PATH}) bulunamadı!")
        print("Lütfen önce train.py dosyasını çalıştırarak modeli eğitin.")
        exit(1)

    print("Kelebek Sınıflandırıcı uygulaması başlatılıyor...")
    demo.launch(share=True)