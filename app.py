import streamlit as st
import pandas as pd
import mediapipe as mp
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2 
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------------------------------------------
# INITIALIZE MEDIAPIPE (New High-Accuracy Tools)
# -----------------------------------------------------------------
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# -----------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# -----------------------------------------------------------------
CNN_LABEL_MAP = {
    0: 'acne',
    1: 'dry',
    2: 'normal',
    3: 'oily',
    4: 'wrinkles' 
}

CONFIDENCE_THRESHOLD = 40.0

# -----------------------------------------------------------------
# LOADING ASSETS (Optimized for 8GB RAM)
# -----------------------------------------------------------------
@st.cache_resource
def load_models_and_assets():
    st.write("Loading AI models... This may take a moment.")
    
    # 1. Load CNN Model
    try:
        cnn_model = tf.keras.models.load_model('models/cnn_model.h5')
    except Exception as e:
        st.error(f"Error loading cnn_model.h5: {e}")
        return None, None

    # 2. Load Product Database
    try:
        df = pd.read_csv("products_with_scores.csv")
    except FileNotFoundError:
        st.error("Error: 'products_with_scores.csv' not found. Run precompute_scores.py first.")
        return None, None
    
    # NOTE: Haar Cascade removed to save memory since we use MediaPipe now
    st.success("System Ready!")
    return cnn_model, df

# -----------------------------------------------------------------
# IMAGE PROCESSING & ZONE EXTRACTION
# -----------------------------------------------------------------
def extract_skin_zones(img_array):
    """
    Identifies and crops specific skin zones: Forehead and Cheeks.
    """
    results = face_mesh.process(img_array)
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = img_array.shape

    def get_ptr(idx):
        return int(landmarks[idx].x * w), int(landmarks[idx].y * h)

    # Zone cropping logic (using lowercase keys for consistency)
    fh_x, fh_y = get_ptr(10)
    forehead = img_array[max(0, fh_y-50):fh_y+50, max(0, fh_x-50):fh_x+50]

    lc_x, lc_y = get_ptr(234)
    left_cheek = img_array[max(0, lc_y-50):lc_y+50, max(0, lc_x-50):lc_x+50]

    rc_x, rc_y = get_ptr(454)
    right_cheek = img_array[max(0, rc_y-50):rc_y+50, max(0, rc_x-50):rc_x+50]

    return {"forehead": forehead, "left_cheek": left_cheek, "right_cheek": right_cheek}

def process_and_validate_image(image):
    """
    Uses MediaPipe for robust face detection and cropping.
    """
    img_array = np.array(image.convert('RGB'))
    img_h, img_w, _ = img_array.shape
    
    results = face_detection.process(img_array)
    
    if not results.detections:
        return False, "No face detected. Ensure your face is clearly visible.", img_array, None

    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    
    x = int(bbox.xmin * img_w)
    y = int(bbox.ymin * img_h)
    w = int(bbox.width * img_w)
    h = int(bbox.height * img_h)

    # Face proximity validation
    face_ratio = h / img_h
    if face_ratio < 0.15:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 3)
        return False, "Move closer to the camera.", img_array, None

    # Success visualization
    cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 3)

    pad_w, pad_h = int(w * 0.2), int(h * 0.2)
    y1, y2 = max(0, y - pad_h), min(img_h, y + h + pad_h)
    x1, x2 = max(0, x - pad_w), min(img_w, x + w + pad_w)
    
    cropped_face = np.array(image.convert('RGB'))[y1:y2, x1:x2]
    
    return True, "Face verified.", img_array, cropped_face

def predict_skin_type(cnn_model, face_image, label_map):
    img_tensor = tf.convert_to_tensor(face_image, dtype=tf.float32)
    img_resized = tf.image.resize(img_tensor, (224, 224))
    img_preprocessed = preprocess_input(img_resized)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    
    prediction = cnn_model.predict(img_batch)
    probabilities = prediction[0]
    predicted_index = np.argmax(probabilities)
    confidence = probabilities[predicted_index] * 100
    
    if confidence < CONFIDENCE_THRESHOLD:
        return "uncertain", confidence, probabilities
        
    return label_map.get(predicted_index, "Unknown"), confidence, probabilities

def get_recommendations(product_df, predicted_skin_type):
    if predicted_skin_type == "uncertain":
        return pd.DataFrame()
    score_column = f"score_{predicted_skin_type}"
    if score_column not in product_df.columns:
        return pd.DataFrame()
    recommended_df = product_df[product_df[score_column] > 0.5]
    return recommended_df.sort_values(by=score_column, ascending=False).head(5)

# -----------------------------------------------------------------
# MAIN APP (Integrated Tabs & Multi-Zone Display)
# -----------------------------------------------------------------
def main():
    st.set_page_config(page_title="DermaClust AI", layout="centered")
    st.title("✨ DermaClust: Advanced Skin Analysis")
    
    cnn_model, df = load_models_and_assets()
    if not all([cnn_model, df is not None]): st.stop()

    # Restoration of Tab Feature
    tab1, tab2 = st.tabs(["📷 Upload Photo", "📸 Use Camera"])
    image_to_process = None
    
    with tab1:
        uploaded = st.file_uploader("Choose a clear selfie", type=["jpg", "png", "jpeg"])
        if uploaded: image_to_process = Image.open(uploaded)
            
    with tab2:
        camera_photo = st.camera_input("Take a live selfie")
        if camera_photo: image_to_process = Image.open(camera_photo)

    if image_to_process:
        st.divider()
        
        # 1. Verification Step
        with st.spinner("Analyzing face geometry..."):
            is_valid, msg, boxed_img, cropped_face = process_and_validate_image(image_to_process)
        
        st.image(boxed_img, caption="Verification View", width=450)
        
        if not is_valid:
            st.error(f"⚠️ {msg}")
        else:
            # 2. Zone Extraction & Display
            st.subheader("🔍 Localized Skin Analysis")
            st.info("Extracting high-detail patches for precise texture assessment.")
            
            # Use original clean array for zone extraction
            clean_array = np.array(image_to_process.convert('RGB'))
            zones = extract_skin_zones(clean_array)
            
            if zones:
                # Display zones side-by-side using columns
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.image(zones["forehead"], caption="Forehead", use_column_width=True)
                with col2:
                    st.image(zones["left_cheek"], caption="Left Cheek", use_column_width=True)
                with col3:
                    st.image(zones["right_cheek"], caption="Right Cheek", use_column_width=True)
                
                st.success("Zones extracted. Analyzing patterns...")
            else:
                st.warning("Could not isolate skin zones. Ensure full face is visible.")

            # 3. Predict & Recommend
            skin_type, confidence, all_probs = predict_skin_type(cnn_model, cropped_face, CNN_LABEL_MAP)
            
            if skin_type != "uncertain":
                st.metric("Detected Skin Type", skin_type.capitalize(), f"{confidence:.1f}% Confidence")
                
                st.divider()
                st.subheader(f"🧴 Top Recommendations for {skin_type.capitalize()}")
                results = get_recommendations(df, skin_type)
                
                if not results.empty:
                    for _, row in results.iterrows():
                        with st.container():
                            st.markdown(f"**{row['product_name']}**")
                            st.caption(f"{row['product_type']} | ${row['product_price']}")
                            with st.expander("View Active Ingredients"):
                                st.write(row['clean_ingreds'])
                            st.divider()
                else:
                    st.info("No products matching this skin profile were found.")
            else:
                st.warning("Analysis uncertain. Please try again with better lighting.")

if __name__ == "__main__":
    main()