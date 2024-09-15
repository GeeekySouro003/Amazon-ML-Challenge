# ML Challenge: Feature Extraction from Images

## Problem Statement

**Q: What is the objective of this hackathon?**

A: The goal is to develop a machine learning model that extracts key product details (e.g., weight, volume, voltage, wattage, dimensions) from images. This is important for digital marketplaces where product information is often incomplete. The model should accurately identify and classify these details directly from images.

## Data Description

**Q: What does the dataset contain?**

A: The dataset includes:

1. **index:** Unique identifier for each sample.
2. **image_link:** Public URL for downloading the product image.
3. **group_id:** Product category code.
4. **entity_name:** The name of the product entity (e.g., "item_weight").
5. **entity_value:** The value of the product entity (e.g., "34 gram"). Note: The test set will not include `entity_value`.

## Output Format

**Q: How should the output be formatted?**

A: The output CSV should have:

1. **index:** The unique identifier matching the test data.
2. **prediction:** A string in the format "x unit" where `x` is a float and `unit` is one of the allowed units (e.g., "2 gram", "12.5 centimetre"). If no value is found, use an empty string.

## Files

**Q: What are the key files provided?**

1. **src/sanity.py:** Ensures output file formatting.
2. **src/utils.py:** Contains functions for downloading images.
3. **src/constants.py:** Lists allowed units.
4. **dataset/train.csv:** Training data with labels.
5. **dataset/test.csv:** Test data without labels.
6. **dataset/sample_test.csv:** Sample input for testing.
7. **dataset/sample_test_out.csv:** Sample output file format.

## Constraints

**Q: What should be considered for submission?**

- The output file must match the format of `sample_test_out.csv` and pass the sanity check.
- Use only the units listed in `constants.py` and the Appendix.

## Evaluation

**Q: How will submissions be evaluated?**

A: Evaluation is based on the F1 score, which is calculated as:

- **True Positives:** Correct predictions with non-empty outputs.
- **False Positives:** Incorrect predictions with non-empty outputs.
- **False Negatives:** Missed predictions with empty outputs.
- **Precision:** True Positives / (True Positives + False Positives)
- **Recall:** True Positives / (True Positives + False Negatives)
- **F1 Score:** 2 * (Precision * Recall) / (Precision + Recall)

## Submission

**Q: How should the final submission be made?**

A: Submit a `test_out.csv` file with the exact format as `sample_test_out.csv` via the Portal.

## Appendix

Allowed units for each entity are listed in `constants.py` and summarized below:

- **Width, Depth, Height:** centimetre, foot, millimetre, metre, inch, yard
- **Item Weight, Max Weight Recommendation:** milligram, kilogram, microgram, gram, ounce, ton, pound
- **Voltage:** millivolt, kilovolt, volt
- **Wattage:** kilowatt, watt
- **Volume:** cubic foot, microlitre, cup, fluid ounce, centilitre, imperial gallon, pint, decilitre, litre, millilitre, quart, cubic inch, gallon


## Approach

### Data Preparation

- **Dataset:** The project utilizes a dataset with images and corresponding labels for training, which annotate product features. A separate test set, with images but without labels, is used to evaluate model performance under real-world conditions.
- **Image Downloading:** Images are downloaded from provided URLs using the `download_images` function. If a download fails, placeholder images are used to maintain dataset continuity.
- **Preprocessing:** Images are resized to 224x224 pixels and normalized to standardize inputs for the model. This ensures consistency in image format for both training and inference.

### Machine Learning Models

#### Feature Extraction Model

- **Convolutional Neural Network (CNN):** 
  - **Convolutional Layers:** For automatic feature detection.
  - **Activation Functions:** Such as ReLU to introduce non-linearity.
  - **Pooling Layers:** To reduce spatial dimensions and computational complexity.
  - **Fully Connected Layers:** To synthesize features into meaningful classifications or regressions.
  
  The CNN model learns and extracts relevant visual features used to predict key product characteristics.

### Training Process

- **Data Loaders:** Custom data loaders handle image loading and preprocessing. The `get_data_loaders` function ensures proper batching and shuffling of both training and validation datasets, maintaining data integrity and enabling efficient training.
- **Model Training:**
  - **Loss Function:** CrossEntropyLoss for classification tasks.
  - **Optimizer:** Adam, for updating model weights based on loss gradients.
  - **Epochs:** Multiple training epochs to iteratively minimize loss on the training set and monitor performance on the validation set to prevent overfitting.

### Experiments

#### Model Evaluation

- **Accuracy Metrics:** Performance is evaluated using:
  - **Accuracy:** Proportion of correct predictions.
  - **Loss:** Model's error during training and validation.
  - **Precision, Recall, and F1 Score:** Comprehensive assessment of classification ability, especially with imbalanced data.

- **Predictions:** The trained model is applied to the test dataset to generate predictions for each image. The `predict.py` script automates this process, outputting predictions in a structured format (e.g., "2 grams" for weight).

## Conclusion

The project demonstrates the successful application of convolutional neural networks (CNNs) for extracting product features directly from images. Key conclusions include:

- **Model Performance:** The CNN model achieved high accuracy and low loss on both training and validation datasets.
- **Feature Extraction:** The model reliably identified and classified product attributes based solely on image data.
- **Challenges:** Issues such as diverse image qualities and consistent preprocessing were addressed with robust data handling techniques and preprocessing pipelines.

### Future Work

- **Model Robustness:** Enhancing robustness to handle a wider variety of image conditions (e.g., low resolution or varying angles).
- **Advanced Architectures:** Incorporating transfer learning to further improve accuracy and efficiency.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Any relevant acknowledgements or credits]

## Contact

For any questions or further information, please contact souradip.eth@gmail.com.