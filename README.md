# Brain Tumor MRI Image Classification

This project uses deep learning (custom CNN and ResNet50 transfer learning) for multi-class classification of brain MRI images (glioma, meningioma, pituitary, no tumor). It includes complete code for training, evaluation, comparison, and deployment with a Streamlit web app.

---

## Project Overview

- Classification of brain MRI images into four classes for early tumor detection.
- Implemented and compared a custom Convolutional Neural Network (CNN) and a pre-trained ResNet50 model.
- Includes evaluation, visualizations (confusion matrices, accuracy/loss curves), and business impact discussion.
- Streamlit web app: User can upload an MRI and get instant predictions.

## Directory Structure
/Brain-Tumor-MRI-Classification/

│

├── train_models.py

├── predict_and_save.py

├── evaluate_and_compare.py

├── app.py

├── /images (charts, screenshots)

├── Brain_Tumor_Model_Comparison.xlsx

├── README.md
## Usage

1. **Clone this repository:**
    ```
    git clone [Your GitHub Repo Link]
    cd Brain-Tumor-MRI-Classification
    ```

3. **Train or Load Models:**
    - Run `train_models.py` to train models from scratch, or download pre-trained weights:

    > **Custom CNN model weights:** [Model Download Link from Google Drive]

4. **Evaluate Models:**
    ```
    python evaluate_and_compare.py
    ```
    - Generates performance metrics and confusion matrices

5. **Run Streamlit Web App:**
    ```
    streamlit run app.py
    ```
    - Open your browser at `http://localhost:8501`
    - Upload MRI images and see predictions and confidence

## Results

- **Custom CNN:** Test Accuracy = 90.24%
- **ResNet50:** Test Accuracy = 67.07%
- Visualizations included: confusion matrices, accuracy/loss graphs
- Streamlit app demo:  
    ![App Screenshot](images/streamlit_app_demo.png)

## Dataset

- Used a public dataset ([add Kaggle or dataset URL/source]) with labeled MRI brain scans.

## Model Weights

- Trained `.h5` model file is too large for GitHub direct upload.
- Download from Google Drive: [Your Google Drive Link]

## Project Report

- Full report available in `Project_Report.pdf` or `docs/`

## Author

- [Your Name]
- [Contact details, optional]

## License

MIT License

## Acknowledgments

- Inspired by public datasets and Kaggle challenges.
- Special thanks to professors and collaborators.
