Overview
This project is a deep learning-powered brain tumor detection platform designed to assist in early diagnosis and classification of brain tumors from MRI scans. Leveraging a hybrid CNN model combining VGG16 and ResNet-50, along with EfficientNetB2 and Grad-CAM visualization, the system provides accurate predictions with interpretable insights. An interactive web interface built using Streamlit enables users to upload scans, view classification results, and generate PDF reports with personalized information.

Key Features
Hybrid Deep Learning Model: Combines VGG16 and ResNet-50 for enhanced feature extraction and classification performance.

Explainable AI: Integrates Grad-CAM to highlight tumor regions in the MRI scan for interpretability.

User-Friendly Interface: A clean, intuitive Streamlit app for image upload, result viewing, and PDF report generation.

Symptom-Based Screening: Optional user input allows preliminary assessment based on symptoms.

Custom PDF Reports: Automatically generates downloadable reports containing prediction, Grad-CAM output, patient name, and ID.

Technology Stack
Deep Learning Models: VGG16, ResNet-50, EfficientNetB2

Explainability: Grad-CAM (Gradient-weighted Class Activation Mapping)

Frontend: Streamlit

Backend: Python, TensorFlow/Keras, OpenCV

PDF Generation: ReportLab / FPDF / PyMuPDF

How It Works
User uploads a contrast-enhanced MRI scan via the web interface.

The scan is preprocessed and passed through the hybrid model for classification (e.g., Glioma, Meningioma, Pituitary, or No Tumor).

Grad-CAM highlights the region influencing the model's decision.

A downloadable PDF report is generated with results, visualizations, and user details.

Users can also optionally fill a symptom form for early screening insights.

Setup & Installation
Clone the repository.

Install Python dependencies listed in requirements.txt.

Download the pre-trained models and place them in the appropriate directory.

Run the Streamlit app using streamlit run app.py.

Access the app via localhost and start testing with sample MRI images.

For inquiries or collaborations, please contact ashrithsambaraju@gmail.com

