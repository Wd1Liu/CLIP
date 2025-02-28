import os
import torch
import clip
from PIL import Image

"""
This script calculates an alignment score for images using CLIP.
The text prompt is derived solely from the parent folder name (e.g., "A_brown_teddy_bear_on_a_bed" -> "A brown teddy bear on a bed"),
and the alignment category is obtained from the subfolder name (e.g., "3-Low_Align", "3-Mid_Align").
Each image's score is computed as the cosine similarity between its CLIP image features and the text prompt,
transformed to a 0–100 range. The alignment category is printed for reference.
"""

# 1) Select the device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2) Load the pre-trained CLIP model and its preprocessing function
model, preprocess = clip.load("ViT-B/32", device=device)

# 3) Define the parent folder path containing your images
# Example: Parent folder "A_brown_teddy_bear_on_a_bed" contains subfolders like "3-Low_Align", "3-Mid_Align", etc.
parent_folder_path = r"C:\Users\lpyst\Desktop\25spring\vip cvgm\CLIP\dataset_test\A_brown_teddy_bear_on_a_bed"

# 4) Derive the base text prompt from the parent folder name (without alignment info)
# e.g., "A_brown_teddy_bear_on_a_bed" -> "A brown teddy bear on a bed"
base_prompt = os.path.basename(parent_folder_path).replace("_", " ")
text_prompt = f"a photo of {base_prompt}"  # Text prompt does NOT include alignment information

# Tokenize the text prompt (remains constant for all images in the parent folder)
text_input = clip.tokenize([text_prompt]).to(device)

# 5) Iterate over each subfolder in the parent folder
for subfolder in os.listdir(parent_folder_path):
    subfolder_path = os.path.join(parent_folder_path, subfolder)
    if not os.path.isdir(subfolder_path):
        continue  # Skip non-directory items

    # Use subfolder name as the alignment category (e.g., "3-Low_Align")
    alignment_category = subfolder

    # List all PNG images in the subfolder
    image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(".png")]

    print(f"\nProcessing subfolder (Alignment Category): {alignment_category}")
    print(f"Text Prompt: '{text_prompt}'\n")

    for image_file in image_files:
        image_path = os.path.join(subfolder_path, image_file)
        # Open and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Encode image and text features using CLIP (without gradient calculation)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            # Use the same tokenized text prompt for all images
            text_features = model.encode_text(text_input)

        # Normalize features to compute cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity between the image and the text prompt
        similarity = image_features @ text_features.T  # shape: [1, 1]
        cos_sim = similarity.item()

        # Transform cosine similarity from [-1, 1] to a 0–100 range
        score = (cos_sim + 1) / 2 * 100

        # Print results with the alignment category indicated
        print(f"Image: {image_file}")
        print(f"Alignment Category: {alignment_category}")
        print(f"Raw Cosine Similarity: {cos_sim:.4f}")
        print(f"Alignment Score [0–100]: {score:.2f}\n")
