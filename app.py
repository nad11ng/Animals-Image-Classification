from pathlib import Path

import gradio as gr

from src.inference import InferenceEngine


CHECKPOINT_PATH = Path("outputs/checkpoints/best.pt")

CLASS_NAMES = [
    "butterfly",
    "cat",
    "chicken",
    "cow",
    "dog",
    "elephant",
    "horse",
    "sheep",
    "spider",
    "squirrel",
]


# The model is loaded only once when the app starts.
inference_engine = InferenceEngine(checkpoint_path=CHECKPOINT_PATH)


def classify_image(image):
    if image is None:
        raise gr.Error("Please upload an image first.")

    try:
        predictions = inference_engine.predict(image=image, top_k=3)
    except Exception as error:
        raise gr.Error(f"Prediction failed: {error}") from error

    return predictions


description = (
    "Upload or drag and drop an animal image. "
    "The model will return its top three predictions.\n\n"
    "Supported classes: "
    + ", ".join(CLASS_NAMES)
    + "."
)


demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Animal image"),
    outputs=gr.Label(num_top_classes=3, label="Top-3 predictions"),
    title="Animal Image Classification",
    description=description,
)


if __name__ == "__main__":
    print("===== STARTING GRADIO APPLICATION =====")
    print(f"Device: {inference_engine.device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print("The model has been loaded and is ready for inference.")

    demo.launch()