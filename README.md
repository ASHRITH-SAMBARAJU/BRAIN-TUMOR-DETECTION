# Brain Tumor Detection & Symptom-Based Recommendation System

## Overview  
This is a deep learning-based web application for early detection and classification of brain tumors using contrast-enhanced MRI scans, with an integrated symptom-based analysis module for initial screening. The system combines a hybrid CNN architecture (VGG16 + ResNet-50), EfficientNetB2, and Grad-CAM for accurate, explainable predictions. The Streamlit interface provides easy upload, real-time classification, and downloadable PDF medical reports.

---

## Key Features  

- **Hybrid CNN Model**: VGG16 and ResNet-50 combined for enhanced feature extraction and tumor classification.  
- **EfficientNetB2 Support**: Lightweight high-performance model integration.  
- **Explainable AI**: Grad-CAM visualizations show the exact tumor region influencing the decision.  
- **Symptom-Based Risk Screening**: Analyzes user-reported symptoms to offer possible tumor types and recommendations.  
- **Automated Medical Reports**: Personalized, downloadable PDF reports including predictions, Grad-CAM maps, and treatment suggestions.  
- **Interactive UI**: Built with Streamlit for seamless user experience.

---

## Technology Stack  

- **Deep Learning Models**: VGG16, ResNet-50, EfficientNetB2  
- **Explainability**: Grad-CAM (Class Activation Mapping)  
- **Frontend**: Streamlit  
- **Backend**: Python, TensorFlow/Keras, OpenCV  
- **PDF Generation**: ReportLab / FPDF / PyMuPDF  

---

## Output Screenshots

### MRI-Based Classification

**Filling Patient Details**  
![Filling Patient Details](data/outputs/filling_patient%20details.jpg)

**MRI-Based Classification Output**  
![MRI Classification](data/outputs/mri_based_classification.jpg)

**MRI-Based Treatment Recommendation**  
![MRI Treatment Recommendation](data/outputs/mri_based_treatment_recommedation.jpg)

**MRI-Based Medical Report**  
![MRI Medical Report](data/outputs/mri_based_medicalreport.png)

---

### Symptom-Based Classification

**Symptom-Based Classification Interface**  
![Symptom Classification](data/outputs/symptom_based_classification.jpg)

**Symptom-Based Treatment Recommendation**  
![Symptom Treatment Recommendation](data/outputs/symptom_based_treatment_recommendation.jpg)

**Symptom-Based Medical Report**  
![Symptom Medical Report](data/outputs/symptom_based_medicalreport.png)

---

### Evaluation Results

**Model Accuracy, Loss, and Classification Report**  
![Evaluation Results](data/outputs/evaluation_results.png)

---

## How to Run the Project

### 1. Clone the Repository and Set Up the Environment

```bash
git clone https://github.com/ASHRITH-SAMBARAJU/BRAIN-TUMOR-DETECTION.git
cd BRAIN-TUMOR-DETECTION

# (Optional) Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate   # For Windows

# Install required dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run main.py



