import torch
import pandas as pd
import os
from model import FeatureExtractionCNN
from data_loader import get_transforms
from src.utils import download_images  # Assuming utility to download images
from src.constants import ALLOWED_UNITS  # Assuming constants for units
from PIL import Image

def load_model(model_path):
    # Initialize and load the trained model
    model = FeatureExtractionCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def predict(image, model, transform):
    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        # Assuming model returns two outputs: entity_value and unit_idx
        predicted_value = output[0].item()  # The predicted value
        predicted_unit_idx = output[1].argmax(dim=1).item()  # Predicted unit index
        predicted_unit = ALLOWED_UNITS[predicted_unit_idx]  # Map index to unit
    return predicted_value, predicted_unit

def format_prediction(value, unit):
    """ Format the prediction as 'x unit' with proper formatting """
    if value is None or unit is None:
        return ""
    
    # Format the prediction string
    return f"{value:.2f} {unit}"

def main():
    # Define paths
    model_path = 'output/cnn_model.pth'
    test_csv_path = 'dataset/test.csv'
    test_image_dir = 'images/test_images/'
    output_csv_path = 'output/test_out.csv'

    # Load the trained model
    model = load_model(model_path)

    # Load transformations for images
    transform = get_transforms()

    # Read test data
    test_data = pd.read_csv(test_csv_path)
    
    predictions = []

    # Iterate over each sample in test data
    for idx, row in test_data.iterrows():
        img_url = row['image_link']
        image_path = os.path.join(test_image_dir, f"{idx}.jpg")
        
        # Download the image if it doesn't exist locally
        if not os.path.exists(image_path):
            download_images(img_url, image_path)
        
        # Open the image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            predictions.append("")
            continue

        # Predict the entity value and unit
        predicted_value, predicted_unit = predict(image, model, transform)

        # Format the prediction and append to the list
        formatted_prediction = format_prediction(predicted_value, predicted_unit)
        predictions.append(formatted_prediction)

    # Create a DataFrame for output
    output_df = pd.DataFrame({
        'index': test_data['index'],
        'prediction': predictions
    })

    # Save predictions to CSV
    output_df.to_csv(output_csv_path, index=False)

    print(f"Predictions saved to {output_csv_path}")

if __name__ == "__main__":
    main()
