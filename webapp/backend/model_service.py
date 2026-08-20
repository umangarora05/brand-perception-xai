import os
import sys
import cv2
import numpy as np
import pandas as pd
import joblib
import shap
import json

# Load models at startup
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Load Differential Model
diff_model_path = os.path.join(MODEL_DIR, 'Differential_Trust_Engine.pkl')
if os.path.exists(diff_model_path):
    model_lgb = joblib.load(diff_model_path)
    explainer = shap.TreeExplainer(model_lgb)
else:
    model_lgb = None
    explainer = None
    print("Warning: Differential_Trust_Engine.pkl not found!")

# Load 167k Corroborating Model
logo_167k_model_path = os.path.join(MODEL_DIR, 'logo_behavior_model.pkl')
if os.path.exists(logo_167k_model_path):
    model_167k = joblib.load(logo_167k_model_path)
else:
    model_167k = None

def process_logo(image_bytes: bytes, filename: str):
    """
    Analyzes a logo image and extracts features, returning trust scores and SHAP impact.
    """
    if model_lgb is None or explainer is None:
        raise ValueError("Machine learning models are not loaded.")

    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Unreadable file format for {filename}")

    # 1. PIXEL INGESTION & NORMALIZATION
    img_resized = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    
    # A. Optimal Ink Extraction
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_pixels = img_rgb[binary_mask > 0]
    
    # B. Spectral Extraction (0.0 - 1.0)
    if len(ink_pixels) > 0:
        r, g, b = np.mean(ink_pixels, axis=0) / 255.0
        sat = np.mean(hsv[binary_mask > 0, 1]) / 255.0
    else:
        r, g, b, sat = 0.0, 0.0, 0.0, 0.0

    # C. Structural Extraction
    edges = cv2.Canny(gray, 50, 150)
    weight = float(np.sum(edges > 0) / (224 * 224))
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circ_list = [4 * np.pi * cv2.contourArea(c) / (cv2.arcLength(c, True)**2) 
                 for c in contours if cv2.arcLength(c, True) > 0]
    circularity = float(np.mean(circ_list)) if circ_list else 0.5
    
    flipped_mask = cv2.flip(binary_mask, 1)
    symmetry = float(np.sum(cv2.bitwise_and(binary_mask, flipped_mask)) / np.sum(binary_mask)) if np.sum(binary_mask) > 0 else 0.5

    # D. Differential features
    occlusion_ratio = float(np.sum(binary_mask > 0) / (224 * 224))
    structural_degradation = 1.0 - symmetry
    neural_displacement = 0.0
    deep_feature_mean = float(np.mean(img_rgb) / 255.0)

    # E. Engineered interaction features
    color_harmony_interaction = (b - r) * symmetry
    structure_density = circularity * (1 - weight)

    # 2. MACHINE LEARNING INFERENCE — all 13 features
    features_list = [
        r, g, b, sat, weight, circularity,
        occlusion_ratio, structural_degradation,
        neural_displacement, symmetry, deep_feature_mean,
        color_harmony_interaction, structure_density
    ]
    
    feature_names = [
        'red', 'green', 'blue', 'saturation',
        'typography_weight', 'font_circularity',
        'occlusion_ratio', 'structural_degradation',
        'neural_displacement', 'harmony_score', 'deep_feature_mean',
        'color_harmony_interaction', 'structure_density'
    ]

    input_data = pd.DataFrame([features_list], columns=feature_names)

    # Primary prediction from differential model
    prediction = float(model_lgb.predict(input_data)[0])
    raw_shap = explainer.shap_values(input_data)
    instance_shap = raw_shap[0].flatten() if isinstance(raw_shap, list) else raw_shap.flatten()

    # Cross-validation with 167k model
    input_data_7 = input_data[['red', 'green', 'blue', 'saturation',
                                'typography_weight', 'font_circularity', 'harmony_score']]
    
    if model_167k is not None:
        prediction_167k = float(model_167k.predict(input_data_7)[0])
        final_prediction = float((prediction * 0.6) + (prediction_167k * 0.4))
    else:
        prediction_167k = None
        final_prediction = float(prediction)

    # --- 3. PATENT INVENTION LOGIC: DYNAMIC SIGNAL CALIBRATION ---
    is_achromatic = sat < 0.05
    
    if is_achromatic:
        instance_shap[0:4] = 0.0
        color_mode = "Achromatic Mode"
        dominant_name = "None (Achromatic)"
    else:
        color_vals = {'red': r, 'green': g, 'blue': b}
        dominant_channel = max(color_vals, key=color_vals.get)
        if color_vals[dominant_channel] > 0.4:
            idx_map = {'red': 0, 'green': 1, 'blue': 2}
            dom_idx = idx_map[dominant_channel]
            if instance_shap[dom_idx] < 0:
                instance_shap[dom_idx] = abs(instance_shap[dom_idx])
        color_mode = f"Chromatic Mode ({dominant_channel.title()} Dominant)"
        dominant_name = dominant_channel.title()

    # Dynamic RGB values extracted from logo ink
    R_val = int(r * 255)
    G_val = int(g * 255)
    B_val = int(b * 255)

    display_features = ['red', 'green', 'blue', 'saturation',
                        'typography_weight', 'font_circularity', 'harmony_score']
    display_shap = [float(x) for x in instance_shap[:7]]

    # Strategy Mappings
    audit_results = pd.DataFrame({'Pillar': display_features, 'Impact': display_shap})
    real_friction = audit_results[audit_results['Impact'] < 0].sort_values(by='Impact')
    
    strategy_mapping = {
        'red': {
            "title": "HIGH CHROMATIC AROUSAL",
            "description": "The red channel is generating spectral friction — an elevated red signal is associated with urgency and aggression, which suppresses consumer trust. Recommended action: Reduce red channel intensity and shift the dominant hue toward blue or neutral tones to move into a calmer trust-manifold."
        },
        'green': {
            "title": "WAVELENGTH IMBALANCE",
            "description": "The green channel is creating a hue imbalance that disrupts spectral harmony. Recommended action: Standardise green channel intensity to complement the primary brand colour — avoid isolated green signals without chromatic context."
        },
        'blue': {
            "title": "SUBOPTIMAL CHROMATIC AUTHORITY",
            "description": "The blue channel is underperforming its trust potential. Blue is the strongest trust signal in consumer psychology, but the current value is not deep enough to activate full authority perception. Recommended action: Deepen the blue hue toward a more saturated, darker tone to maximise brand authority."
        },
        'typography_weight': {
            "title": "COGNITIVE DENSITY OVERLOAD",
            "description": "The stroke weight of the typography is too heavy — dense letterforms create cognitive friction that consumers interpret as visual aggression rather than confidence. Recommended action: Reduce stroke weight to improve clarity."
        },
        'font_circularity': {
            "title": "ANGULAR DISSONANCE",
            "description": "The geometry of the logo contains too many harsh, angular edges. Sharp angles trigger implicit threat responses in the brain, lowering trust. Recommended action: Soften sharp edges or integrate more circular forms to increase perceptual approachability."
        },
        'saturation': {
            "title": "EXCESSIVE SATURATION FATIGUE",
            "description": "The colour saturation is overly intense, causing perceptual fatigue and signaling a lack of premium restraint. Recommended action: Desaturate the primary brand colours slightly to convey sophistication and maturity."
        },
        'harmony_score': {
            "title": "STRUCTURAL ASYMMETRY",
            "description": "The logo lacks horizontal symmetry, creating an unbalanced composition that the brain finds difficult to process smoothly. Recommended action: Adjust internal spacing and alignment to create a more balanced, harmonious structure."
        }
    }

    prescriptions = []
    if not real_friction.empty:
        for _, row in real_friction.iterrows():
            pillar = row['Pillar']
            impact = row['Impact']
            if pillar in strategy_mapping:
                prescriptions.append({
                    "pillar": pillar,
                    "impact": float(impact),
                    "title": strategy_mapping[pillar]["title"],
                    "description": strategy_mapping[pillar]["description"]
                })

    return {
        "identity": filename,
        "engine_status": color_mode,
        "ink_color": {"r": float(r), "g": float(g), "b": float(b), "R": R_val, "G": G_val, "B": B_val},
        "dominant_channel": dominant_name,
        "scores": {
            "differential": prediction,
            "corroborating_167k": prediction_167k,
            "ensemble_trust_coefficient": final_prediction
        },
        "shap": {
            "features": display_features,
            "impacts": display_shap
        },
        "prescriptions": prescriptions,
        "quantitative_report": prescriptions  # same data but used for quantitative display
    }
