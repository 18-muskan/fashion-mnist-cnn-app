
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "fashion_mnist_cnn.keras"

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

model = tf.keras.models.load_model(MODEL_PATH)

def predict_clothing(image):
    if image is None:
        return {}

    image = ImageOps.exif_transpose(image).convert("L")
    image = ImageOps.fit(image, (28, 28))
    pixels = np.array(image, dtype=np.float32)

    if pixels.mean() > 127:
        pixels = 255 - pixels

    pixels = pixels / 255.0
    pixels = pixels.reshape(1, 28, 28, 1)

    probabilities = model.predict(pixels, verbose=0)[0]

    return {
        class_names[i]: float(probabilities[i])
        for i in range(10)
    }

demo = gr.Interface(
    fn=predict_clothing,
    inputs=gr.Image(type="pil", label="Upload a clothing image"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="Fashion MNIST CNN Classifier",
    description="Upload a clothing image to see the model's top predictions."
)

if __name__ == "__main__":
    demo.launch()
