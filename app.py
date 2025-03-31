import os
from flask import Flask, request, jsonify
from diffusers import DiffusionPipeline
from PIL import Image

app = Flask(__name__)

# Load the model
pipeline = DiffusionPipeline.from_pretrained(
    "jadechoghari/mar", trust_remote_code=True, custom_pipeline="jadechoghari/mar"
)

# Ensure output directory exists
OUTPUT_DIR = "./images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route("/generate", methods=["POST"])
def generate_image():
    try:
        # Get parameters from request
        data = request.json
        model_type = data.get("model_type", "mar_huge")
        seed = data.get("seed", 42)
        num_ar_steps = data.get("num_ar_steps", 64)
        class_labels = data.get("class_labels", [207, 360, 388])
        cfg_scale = data.get("cfg_scale", 4)
        cfg_schedule = data.get("cfg_schedule", "constant")

        # Generate the image
        generated_image = pipeline(
            model_type=model_type,
            seed=seed,
            num_ar_steps=num_ar_steps,
            class_labels=class_labels,
            cfg_scale=cfg_scale,
            output_dir=OUTPUT_DIR,
            cfg_schedule=cfg_schedule,
        )

        # Save image
        image_path = os.path.join(OUTPUT_DIR, f"generated_image_{seed}.png")
        generated_image.save(image_path)

        return jsonify({"message": "Image generated successfully", "image_path": image_path})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
