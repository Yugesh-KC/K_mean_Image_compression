from flask import Flask, request, jsonify, send_file
from PIL import Image
import torch
import numpy as np
from io import BytesIO
from flask_cors import CORS
import os  # Import os module to delete files

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def k_means_clustering(image_tensor, num_clusters, max_iters=10):
    pixels = image_tensor.reshape(-1, image_tensor.shape[-1])
    pixels = pixels.float() / 255.0

    # Initialize cluster centers randomly
    cluster_centers = pixels[torch.randperm(pixels.size(0))[:num_clusters]]

    for _ in range(max_iters):
        distances = torch.cdist(pixels, cluster_centers)
        cluster_assignments = torch.argmin(distances, dim=1)

        new_centers = torch.stack([pixels[cluster_assignments == i].mean(dim=0) if (cluster_assignments == i).sum() > 0 else cluster_centers[i] for i in range(num_clusters)])

        if torch.allclose(new_centers, cluster_centers):
            break

        cluster_centers = new_centers

    return cluster_assignments, cluster_centers

def compress_image(image_path, num_clusters, max_iters=100):
    image = Image.open(image_path).convert('RGB')
    image_tensor = torch.from_numpy(np.array(image))

    cluster_assignments, cluster_centers = k_means_clustering(image_tensor, num_clusters, max_iters)

    compressed_image = cluster_centers[cluster_assignments].reshape(image_tensor.shape)
    compressed_image = (compressed_image * 255).byte()

    return Image.fromarray(compressed_image.numpy())

@app.route('/compress', methods=['POST'])
def upload_and_compress():
    try:
        print("Starting image upload and compression process...")

        # Check if 'image' is part of the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image part in the request'}), 400

        file = request.files['image']

        # Extract number of colors and max iterations from the form data
        num_colors = int(request.form.get('num_colors', 2))  # Default to 2 if not provided
        max_iters = int(request.form.get('max_iters', 100))  # Default to 100 if not provided

        # Validate the uploaded file format
        if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
            print("Valid image file detected.")
            image_path = 'uploaded_image.jpg'
            file.save(image_path)

            print("Image saved. Starting compression...")
            compressed_img = compress_image(image_path, num_clusters=num_colors, max_iters=max_iters)

            img_io = BytesIO()
            compressed_img.save(img_io, 'PNG')
            img_io.seek(0)

            # Delete the image file after processing
            os.remove(image_path)  # Deletes the uploaded image file

            return send_file(img_io, mimetype='image/png')
            
        else:
            print("Invalid file format.")
            return jsonify({'error': 'Invalid file format. Please upload a valid image.'}), 400

    except Exception as e:
        print(f"Error during image upload and compression: {e}")
        return jsonify({'error': 'Error during image upload and compression.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
