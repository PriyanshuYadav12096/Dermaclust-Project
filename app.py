import streamlit as st
import pandas as pd
import mediapipe as mp
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2 
import ast
import json
from assessment_logic import calculate_hybrid_profile
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from ingredients import KEY_INGREDIENTS

# -----------------------------------------------------------------
# INITIALIZE MEDIAPIPE
# -----------------------------------------------------------------
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# -----------------------------------------------------------------
# LOADING ASSETS
# -----------------------------------------------------------------
@st.cache_resource
def load_models_and_assets():
    try:
        cnn_model = tf.keras.models.load_model('models/cnn_model.h5')
        df = pd.read_csv("products_with_scores.csv")
        
        # Load the dynamic label map from training
        with open('assets/label_map.json', 'r') as f:
            raw_map = json.load(f)
            # Convert JSON string keys back to integers
            label_map = {int(k): v for k, v in raw_map.items()}
            
        return cnn_model, df, label_map
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None

# -----------------------------------------------------------------
# IMAGE PROCESSING FUNCTIONS
# -----------------------------------------------------------------
def extract_skin_zones(img_array):
    results = face_mesh.process(img_array)
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = img_array.shape
    def get_ptr(idx): return int(landmarks[idx].x * w), int(landmarks[idx].y * h)

    # Use 120x120 patches (patch_size * 2) for better texture sampling
    patch_size = 60 

    fh_x, fh_y = get_ptr(10)
    lc_x, lc_y = get_ptr(234)
    rc_x, rc_y = get_ptr(454)

    return {
        "forehead": img_array[max(0, fh_y-patch_size):fh_y+patch_size, max(0, fh_x-patch_size):fh_x+patch_size],
        "left_cheek": img_array[max(0, lc_y-patch_size):lc_y+patch_size, max(0, lc_x-patch_size):lc_x+patch_size],
        "right_cheek": img_array[max(0, rc_y-patch_size):rc_y+patch_size, max(0, rc_x-patch_size):rc_x+patch_size]
    }

def process_and_validate_image(image):
    img_array = np.array(image.convert('RGB'))
    img_h, img_w, _ = img_array.shape
    results = face_detection.process(img_array)
    
    if not results.detections:
        return False, "No face detected.", img_array, None

    bbox = results.detections[0].location_data.relative_bounding_box
    x, y, w, h = int(bbox.xmin * img_w), int(bbox.ymin * img_h), int(bbox.width * img_w), int(bbox.height * img_h)

    cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    pad_w, pad_h = int(w * 0.2), int(h * 0.2)
    cropped_face = np.array(image.convert('RGB'))[max(0, y-pad_h):min(img_h, y+h+pad_h), max(0, x-pad_w):min(img_w, x+w+pad_w)]
    
    return True, "Face verified.", img_array, cropped_face

# -----------------------------------------------------------------
# AI PREDICTION LOGIC
# -----------------------------------------------------------------
def enhance_skin_texture(img_array):
    img_yuv = cv2.cvtColor(np.array(img_array), cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_output

def predict_skin_type(cnn_model, img, label_map):
    img_enhanced = enhance_skin_texture(img)
    img_resized = tf.image.resize(tf.convert_to_tensor(img_enhanced, dtype=tf.float32), (224, 224))
    img_preprocessed = preprocess_input(img_resized)
    prediction = cnn_model.predict(np.expand_dims(img_preprocessed, axis=0))[0]
    idx = np.argmax(prediction)
    return label_map.get(idx, "Unknown"), prediction[idx] * 100, prediction

def get_hero_ingredients(ingredient_list, skin_type):
    if skin_type not in KEY_INGREDIENTS:
        return []
    beneficial_set = set(KEY_INGREDIENTS[skin_type])
    matches = [ing for ing in ingredient_list if ing.strip().lower() in beneficial_set]
    return matches[:3]

def get_recommendations(product_df, skin_type,user_goal):
    target_col = f"score_{skin_type}"
    if target_col not in product_df.columns:
        target_col = "score_normal" 
    # 1. Start with high-scoring products for the skin type    
    matches = product_df[product_df[target_col] > 0.4].copy()

    # 2. PERSONALIZATION: Boost products matching the user's specific goal
    # We look for goal keywords in the ingredient list or product name
    goal_keywords = {
        "Acne Control": ["salicylic", "benzoyl", "tea tree", "zinc"],
        "Brightening": ["vitamin c", "niacinamide", "glycolic", "kojic"],
        "Anti-Aging": ["retinol", "peptides", "collagen", "adenosine"],
        "Deep Hydration": ["hyaluronic", "glycerin", "ceramide", "squalane"]
    }
    
    if user_goal in goal_keywords:
        keywords = goal_keywords[user_goal]
        # Assign a 'Personalization Boost'
        def calculate_boost(row):
            ingreds = str(row['clean_ingreds']).lower()
            return sum(2.0 for k in keywords if k in ingreds)
        
        matches['boost'] = matches.apply(calculate_boost, axis=1)
        matches[target_col] = matches[target_col] + matches['boost']

    # 3. Sort by the NEW personalized score
    matches = matches.sort_values(by=target_col, ascending=False)
    
    def find_best(category_keywords):
        pattern = '|'.join(category_keywords)
      
    # 1. Try to find a high-scoring match (Score > 0.4)
        best_match = matches[matches['product_type'].str.contains(pattern, case=False, na=False)].head(1)
    
    # 2. FALLBACK: If nothing found, lower the threshold to 0.2
        if best_match.empty:
          lower_matches = product_df[product_df[target_col] > 0.2].copy()
          best_match = lower_matches[lower_matches['product_type'].str.contains(pattern, case=False, na=False)].sort_values(by=target_col, ascending=False).head(1)
    
        return best_match
        # return matches[matches['product_type'].str.contains(pattern, case=False, na=False)].head(1)

    routine = {
        "AM": {
            "Step 1: Cleanse": find_best(['Cleanser', 'Wash']),
            "Step 2: Treat": find_best(['Serum', 'Toner', 'Essence']),
            "Step 3: Protect": find_best(['SPF', 'Sunscreen', 'Day Cream', 'Moisturizer'])
        },
        "PM": {
            "Step 1: Cleanse": find_best(['Cleanser', 'Wash', 'Oil']),
            "Step 2: Treat": find_best(['Serum', 'Treatment', 'Night']),
            "Step 3: Hydrate": find_best(['Moisturizer', 'Cream', 'Night Cream'])
        }
    }
    return routine

# -----------------------------------------------------------------
# MAIN INTERFACE
# -----------------------------------------------------------------
def main():
    st.set_page_config(page_title="DermaClust AI", layout="centered")
    st.title("✨ DermaClust: Professional Skin Analysis")
    
    cnn_model, df, label_map = load_models_and_assets()
    if cnn_model is None or df is None: st.stop()

    tab1, tab2, tab3 = st.tabs(["📷 Upload Photo", "📸 Use Camera", "🛡️ Safety Scanner"])
    img_input = None
    
    with tab1:
        uploaded = st.file_uploader("Upload selfie", type=["jpg", "png", "jpeg"])
        if uploaded: img_input = Image.open(uploaded)
    with tab2:
        camera = st.camera_input("Take selfie")
        if camera: img_input = Image.open(camera)
    
    # Global Clinical Verification UI
    st.divider()
    st.subheader("📋 Clinical Verification")
    col1, col2 = st.columns(2)
    with col1:
        q_feel = st.selectbox("How does your skin feel 1 hour after washing?", 
                              ["Normal", "Tight/Itchy", "Greasy/Shiny", "Oil only in T-Zone"])
    with col2:
        q_breakouts = st.selectbox("How often do you experience breakouts?", 
                                   ["Rarely", "Occasionally", "Frequent"])
    with col2:
    # NEW: Specific Goal for Personalization
        q_goal = st.selectbox(
        "What is your primary skin goal?", 
        ["General Maintenance", "Acne Control", "Brightening", "Anti-Aging", "Deep Hydration"]
    )    
    user_quiz = {'feel_after_wash': q_feel, 'breakouts': q_breakouts,'goal': q_goal}

    final_type = "uncertain"

    if img_input:
        is_valid, msg, boxed_img, face_crop = process_and_validate_image(img_input)
        st.image(boxed_img, caption="Face Detection Status", width=450)
        
        if not is_valid:
            st.error(msg)
        else:
            # --- TARGETED ZONE ANALYSIS (Averaged Logic) ---
            zones = extract_skin_zones(np.array(img_input.convert('RGB')))
            
            if zones:
                st.subheader("🔍 Scientific Skin Patch Analysis")
                cols = st.columns(3)
                all_zone_probs = []

                for i, (name, patch) in enumerate(zones.items()):
                    cols[i].image(patch, caption=name.capitalize(), use_column_width=True)
                    _, _, probs = predict_skin_type(cnn_model, patch, label_map)
                    all_zone_probs.append(probs)
                
                # MATHEMATICAL FIX: Average probabilities to eliminate hair/ear noise
                final_probs = np.mean(all_zone_probs, axis=0)
                ai_results = dict(zip(label_map.values(), final_probs))
                
                # HYBRID MERGE: AI Skin Patches + Clinical User Quiz
                final_type, combined_weights = calculate_hybrid_profile(ai_results, user_quiz)
                
                st.write("### 📊 Verified Skin Profile (AI + Clinical)")
                chart_data = pd.DataFrame(combined_weights.values(), index=combined_weights.keys(), columns=["Combined Score"])
                st.bar_chart(chart_data)

            if final_type != "uncertain":
                st.metric("Detected Profile", final_type.capitalize())
                st.divider()
                
                routine = get_recommendations(df, final_type,user_quiz['goal'])
                st.subheader(f"✨ Recommended {final_type.capitalize()} Regimen")
                
                col_am, col_pm = st.columns(2)
                for period, col in [("AM", col_am), ("PM", col_pm)]:
                    with col:
                        st.markdown(f"#### {'☀️' if period == 'AM' else '🌙'} {period}")
                        for step, prod_df in routine[period].items():
                            if not prod_df.empty:
                                row = prod_df.iloc[0]
                                with st.expander(f"{step}: {row['product_name']}"):
                                    st.caption(f"**Type:** {row['product_type']}")
                                    
                                    ing_list = row['clean_ingreds']
                                    if isinstance(ing_list, str): ing_list = ast.literal_eval(ing_list)
                                    
                                    heroes = get_hero_ingredients(ing_list, final_type)
                                    if heroes:
                                        st.success(f"**Why it works:** Contains **{', '.join(heroes)}**.")
                                    else:
                                        st.write("Matches your profile compatibility score.")
                            else:
                                st.write(f"*{step}: No match found.*")

    with tab3:
        st.subheader("Check a Product You Already Own")
        raw_input = st.text_area("Paste ingredients here (separated by commas or new lines):", height=150)
        if st.button("Run Safety Scan"):
            if final_type == "uncertain":
                st.warning("Please complete the Skin Analysis first.")
            elif not raw_input:
                st.error("Please paste ingredients.")
            else:
                from safety_checker import check_product_safety
                flags = check_product_safety(raw_input, final_type)
                if not flags:
                    st.success(f"✅ Clean Match for {final_type} skin.")
                else:
                    st.error(f"⚠️ Found {len(flags)} red flags:")
                    for f in flags: st.write(f"- **{f['ingredient'].capitalize()}**: {f['reason']}")

if __name__ == "__main__":
    main()