import sys
import os
import tempfile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import cv2
from models.hybrid_vgg16_resnet50 import build_hybrid_model
from utils.gradcam import generate_gradcam
import tensorflow as tf
from fpdf import FPDF
import base64
import re

st.set_page_config(page_title="Brain Tumor Detector", layout="wide")

st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f5f6fa;
            color: #1c1c1c;
        }
        h1 {
            font-size: 2.8rem;
            font-weight: 800;
            color: #2f3640;
            text-align: center;
            margin-bottom: 10px;
        }
        h2 {
            font-size: 1.8rem;
            color: #0097e6;
            margin-top: 40px;
            border-bottom: 1px solid #ccc;
            padding-bottom: 6px;
        }
        .stTabs [role="tab"] {
            background: #dcdde1;
            color: #2f3640;
            font-weight: bold;
            border-radius: 8px 8px 0 0;
            padding: 10px;
            margin-right: 5px;
        }
        .stTabs [aria-selected="true"] {
            background: #00a8ff;
            color: white;
        }
        .stButton > button {
            background-color: #273c75;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 16px;
        }
        .stButton > button:hover {
            background-color: #192a56;
        }
        .image-row {
            display: flex;
            justify-content: space-around;
            align-items: center;
            gap: 96px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("\U0001F9E0 Brain Tumor Detection & Symptom-Based Recommendation System")

model = build_hybrid_model()
model.load_weights("hybrid_model_weights.h5")
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

def generate_pdf(tumor, symptom_tumor, treatment_text, patient_name, patient_id):
    from fpdf import FPDF
    import tempfile, base64
    from datetime import datetime

    logo_path = r"C:\Users\ashri\OneDrive\Desktop\braintumordet\webapp\logo.png"

    class PDF(FPDF):
        def header(self):
            # Logo
            try:
                self.image(logo_path, x=10, y=8, w=18)
            except:
                pass
            self.set_font("Arial", 'B', 16)
            self.set_text_color(0, 70, 140)
            self.cell(0, 10, "MEDICAL REPORT", border=False, ln=True, align='C')
            self.ln(10)

    pdf = PDF()
    pdf.add_page()

    # SECTION HEADER: PATIENT
    pdf.set_fill_color(0, 0, 0)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "SECTION 1: PATIENT'S PARTICULARS", ln=True, fill=True)

    # Field: Patient Name
    pdf.set_text_color(0)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(60, 10, "Full name of patient:", border=1)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 10, remove_emojis(patient_name), border=1, ln=True)

    # Field: Patient ID
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(60, 10, "Patient ID:", border=1)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 10, remove_emojis(patient_id), border=1, ln=True)

    # Field: Tumor Prediction
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(60, 10, "Tumor Prediction:", border=1)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 10, remove_emojis(tumor.capitalize()), border=1, ln=True)

    # Optional Field: Symptom Diagnosis
    if symptom_tumor:
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(60, 10, "Symptom Diagnosis:", border=1)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 10, remove_emojis(symptom_tumor.capitalize()), border=1, ln=True)

    # SECTION HEADER: TREATMENT
    pdf.ln(4)
    pdf.set_fill_color(0, 0, 0)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "SECTION 2: TREATMENT RECOMMENDATION", ln=True, fill=True)

    # Treatment Block (multi-line)
    pdf.set_text_color(0)
    pdf.set_font("Arial", '', 11)
    clean_text = remove_emojis(treatment_text)
    pdf.multi_cell(0, 8, clean_text, border=1)

    # Footer
    pdf.set_y(-20)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(100)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 0, 'C')

    # Save and return
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        pdf.output(temp_file.name)
        with open(temp_file.name, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    return base64_pdf


treatment_map = {
    "glioma": """Immediate neuro-oncological evaluation is recommended.
- Next Steps: MRI with contrast, biopsy confirmation.
- Treatment: Surgery (if operable), followed by radiation/chemotherapy.
- Learn more: https://www.cancer.gov/types/brain/patient/adult-glioblastoma-treatment-pdq""",
    "meningioma": """Meningiomas are often slow-growing.
- Next Steps: Regular MRI scans to monitor size.
- Treatment: Observation, or surgery if symptomatic.
- Learn more: https://www.aans.org/en/Patients/Neurosurgical-Conditions-and-Treatments/Meningioma""",
    "pituitary": """Pituitary tumors may impact hormone regulation.
- Next Steps: Endocrinological blood tests, brain MRI.
- Treatment: Hormone therapy or surgery.
- Learn more: https://www.pituitary.org/""",
    "notumor": """No Tumor Detected.
- Recommendation: Maintain routine checkups and monitor any symptoms."""
}

symptom_db = {
    "Persistent headache": ("glioma", 2),
    "Seizures": ("glioma", 3),
    "Blurred vision": ("pituitary", 2),
    "Memory issues": ("meningioma", 2),
    "Difficulty speaking": ("glioma", 2),
    "Loss of balance": ("meningioma", 3),
    "Nausea": ("pituitary", 1),
    "Weakness in limbs": ("glioma", 3),
    "Menstrual irregularities": ("pituitary", 2),
    "Hearing problems": ("meningioma", 1)
}

def get_timeline(tumor_type):
    return {
        "glioma": [
            "Step 1: MRI with contrast",
            "Step 2: Biopsy for confirmation",
            "Step 3: Surgery (if operable)",
            "Step 4: Radiation + Chemotherapy",
            "Step 5: Regular follow-up imaging"
        ],
        "meningioma": [
            "Step 1: MRI detection",
            "Step 2: Monitor size if small",
            "Step 3: Surgery if growing/symptomatic",
            "Step 4: Yearly MRI check"
        ],
        "pituitary": [
            "Step 1: Hormone testing",
            "Step 2: Pituitary MRI",
            "Step 3: Hormone-suppressive meds",
            "Step 4: Surgery if unresponsive or compressive"
        ],
        "notumor": [
            "Step 1: Maintain routine checkups",
            "Step 2: Monitor symptoms",
            "Step 3: Return if symptoms increase"
        ]
    }.get(tumor_type, [])

# ---- UI Tabs ----
tab1, tab2 = st.tabs(["🧠 MRI Detection", "🧰 Symptom-Based Analysis"])

with tab1:
    st.header("📸 MRI Image Upload & Detection")
    with st.form("patient_info_form1"):
        name = st.text_input("👤 Patient Name")
        pid = st.text_input("🆔 Patient ID")
        submitted = st.form_submit_button("Submit Info")
    if not name or not pid:
        st.warning("⚠️ Please enter both patient name and ID.")
        st.stop()

    file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])
    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        img = cv2.resize(img, (224, 224))
        img_norm = img / 255.0

        st.markdown('<div class="image-row">', unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.image(img, caption="Uploaded MRI", width=300)

        pred = model.predict(np.expand_dims(img_norm, axis=0))
        raw_pred = class_names[np.argmax(pred)]
        tumor_pred = raw_pred.replace('_tumor', '')
        st.success(f"🧠 Predicted Tumor Type: **{tumor_pred.capitalize()}**")

        if st.checkbox("🔍 Show Grad-CAM"):
            heatmap = generate_gradcam(model, img_norm)
            with col2:
                st.image(heatmap, caption="Grad-CAM Heatmap", width=300)

        st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("🎯 MRI-Based Treatment Recommendation")
        treatment = treatment_map.get(tumor_pred, "⚠️ No treatment information available.")
        st.markdown(treatment, unsafe_allow_html=True)
        st.subheader("🗓 Suggested Treatment Timeline")
        for step in get_timeline(tumor_pred):
            st.markdown(f"- {step}")

        base64_pdf = generate_pdf(tumor_pred, None, treatment_map[tumor_pred], name, pid)
        st.download_button("📄 Download PDF Report", base64.b64decode(base64_pdf), "tumor_report.pdf")

with tab2:
    st.header("🧰 Symptom-Based Assessment & Recommendation")
    with st.form("patient_info_form2"):
        name2 = st.text_input("👤 Patient Name", key="symptom_name")
        pid2 = st.text_input("🆔 Patient ID", key="symptom_pid")
        submitted2 = st.form_submit_button("Submit Info")
    if not name2 or not pid2:
        st.warning("⚠️ Please enter both patient name and ID.")
        st.stop()

    symptom_list = list(symptom_db.keys())
    selected_symptoms = st.multiselect("Select your symptoms:", symptom_list)

    if st.button("🧪 Analyze Symptoms"):
        if not selected_symptoms:
            st.info("✅ No symptoms selected.")
        else:
            scores = {"glioma": 0, "meningioma": 0, "pituitary": 0}
            for symptom in selected_symptoms:
                tumor, intensity = symptom_db[symptom]
                scores[tumor] += intensity

            likely = max(scores, key=scores.get)
            total_score = scores[likely]

            st.subheader(f"🧠 Likely Tumor Type from Symptoms: **{likely.capitalize()}**")
            st.write(f"🧪 Symptom Intensity Score: **{total_score}**")

            if total_score >= 6:
                st.warning("⚠️ High symptom intensity. Urgent screening recommended.")
            elif total_score >= 3:
                st.info("🔍 Moderate symptoms. Consider seeing a specialist.")
            else:
                st.success("✅ Mild symptoms. Monitor for changes.")

            st.subheader("🌟 Symptom-Based Treatment Recommendation")
            st.markdown(treatment_map[likely], unsafe_allow_html=True)

            st.subheader("🗓 Suggested Treatment Timeline")
            for step in get_timeline(likely):
                st.markdown(f"- {step}")

            base64_pdf = generate_pdf("N/A", likely, treatment_map[likely], name2, pid2)
            st.download_button("📄 Download PDF Report (Symptoms)", base64.b64decode(base64_pdf), "symptom_report.pdf")

st.markdown("---")
st.write("👋 Thanks for using the Brain Tumor Symptom Analyzer!")