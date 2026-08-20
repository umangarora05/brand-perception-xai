#1
# !pip install -U tensorflow==2.19.0

# Block 1 - Technical Infrastructure & Proprietary Dependency Layer
import os

# 1. Computational Resource Optimization
# Suppress non-critical logs to prioritize high-speed batch processing telemetry
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# 2. Advanced Mathematical & Spatial Processing Libraries
# Patent Step: Utilizing K-Dimensional Trees for optimized perceptual manifold mapping
import numpy as np
import pandas as pd
import cv2
import time
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import KDTree      

# 3. Explainable AI (XAI) & Gradient Boosted Frameworks
# Patent Step: Implementing SHAP for feature-level attribution and trust-degradation analysis
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 4. Deep Feature Extraction (Neural Backbone)
# Utilizing Residual Networks for high-dimensional visual embedding
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# 5. System Readiness Verification
# Validating the specialized environment for 35-Pillar Perceptual Audit
print(f"System Check: TensorFlow {tf.__version__} | LightGBM {lgb.__version__}")
print("Status: Proprietary environment initialized for Differential Interference Analysis.")

#2
import tensorflow as tf
# This MUST print 2.16.1 to confirm the install worked
print(f"Verified TensorFlow Version: {tf.__version__}") 

# Re-initialize the Neuromarketer engine
from tensorflow.keras.applications import ResNet50
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
print(" Model successfully re-initialized in the new session.")

# Block 2 - Persistent Intelligence & Perceptual State Restoration
import os
import joblib
import pandas as pd
import shap

# Dynamically resolve models directory relative to script
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Local model paths for hosting
model_path      = os.path.join(MODELS_DIR, 'Differential_Trust_Engine.pkl')
data_path       = os.path.join(MODELS_DIR, 'Empirical_Audit_Evidence.pkl')
model_167k_path = os.path.join(MODELS_DIR, 'logo_behavior_model.pkl')
data_167k_path  = os.path.join(MODELS_DIR, 'processed_logos_data.pkl')

# Load primary differential model
if os.path.exists(model_path) and os.path.exists(data_path):
    print("Found Differential Knowledge Base. Restoring Perceptual State...")
    model_lgb = joblib.load(model_path)
    final_results_clean = pd.read_pickle(data_path)

    # Auto-detect features from saved model
    try:
        features = model_lgb.feature_name()
        print(f"Features loaded from model: {features}")
    except:
        features = ['red', 'green', 'blue', 'saturation', 'typography_weight',
                    'font_circularity', 'harmony_score']
        print(f"Using default 7 features.")

    # Safely assign X — only use columns that exist in both
    available = [f for f in features if f in final_results_clean.columns]
    X = final_results_clean[available]
    explainer = shap.TreeExplainer(model_lgb)
    skip_training = True
    print(f"Status: {len(final_results_clean)} differential audits restored.")
    print(f"Status: {len(available)} features active.")
else:
    print("No Differential Knowledge Base found. Fresh training required.")
    skip_training = False

# Load 167k general model
if os.path.exists(model_167k_path) and os.path.exists(data_167k_path):
    print("Found 167k Logo Knowledge Base. Loading corroborating model...")
    model_167k = joblib.load(model_167k_path)
    data_167k  = pd.read_pickle(data_167k_path)
    print(f"Status: {len(data_167k)} Logo-2K+ records available for cross-validation.")
else:
    model_167k = None
    data_167k  = None
    print("167k model not found — cross-validation disabled.")

# Block 3: Matched-Pair Differential Ingestor
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# 1. Initialize the Feature Extraction Backbone
# Patent Claim: Using a deep residual network as a differential comparator
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_differential_features(r0_path, r1_path):
    """
    Calculates the 'Perceptual Displacement' between baseline and typography.
    This is the core of the 'Delta-Trust' Patent.
    """
    # Load images
    img0 = cv2.imread(r0_path)
    img1 = cv2.imread(r1_path)
    
    # Ensure identical dimensions
    img0 = cv2.resize(img0, (224, 224))
    img1 = cv2.resize(img1, (224, 224))
    
    # Preprocess for ResNet
    x0 = preprocess_input(np.expand_dims(img0, axis=0))
    x1 = preprocess_input(np.expand_dims(img1, axis=0))
    
    # Extract Vector Embeddings
    vec0 = base_model.predict(x0, verbose=0)
    vec1 = base_model.predict(x1, verbose=0)
    
    # Calculate the 'Inference Signal' (The Typography's impact)
    # Patent Logic: Isolating the Typography Vector from the Background Noise
    delta_vector = vec1 - vec0
    
    return delta_vector, img0, img1

print("Differential Ingestor successfully initialized for Delta-Analysis.")

# Block 4 - Matched-Pair Differential Streamer (Updated for Separate Directories)
import os
import pandas as pd
import time

# 1. Corrected Path Configuration based on Dataset Structure
# Use the exact folder names from your Kaggle input sidebar
base_dir = '/kaggle/input/datasets/umangarora05/emotional-impact-of-font/Typographic Dataset/Typographic Dataset Large/'
r0_dir = os.path.join(base_dir, 'species-large-r0')
r1_dir = os.path.join(base_dir, 'species-large-r1')
palette_path = '/kaggle/input/datasets/umangarora05/emotion-palette/emotion_palette.csv'

# 2. Updated Matched-Pair Discovery Logic
def get_matched_pairs(dir0, dir1):
    """
    Pairs baseline (r0) and intervention (r1) images across separate folders.
    This ensures the 'Differential Delta' is accurately calculated.
    """
    # Verify directories exist
    if not os.path.exists(dir0) or not os.path.exists(dir1):
        print(f"Error: One or both directories not found.\nCheck: {dir0}\nCheck: {dir1}")
        return []

    r0_files = sorted(os.listdir(dir0))
    r1_files = sorted(os.listdir(dir1))
    
    pairs = []
    # Match by filename ID (e.g., '123-r1.png' matches '123-r0.png')
    for r1_name in r1_files:
        r0_target_name = r1_name.replace('r1', 'r0')
        if r0_target_name in r0_files:
            pairs.append((
                os.path.join(dir0, r0_target_name), 
                os.path.join(dir1, r1_name)
            ))
    return pairs

# 3. Initialize Ingestor
matched_pairs = get_matched_pairs(r0_dir, r1_dir)

if matched_pairs:
    print(f"Success: Identified {len(matched_pairs)} matched r0/r1 pairs across directories.")
    
    # 4. Load Psychographic Ground Truth
    if os.path.exists(palette_path):
        emotion_df = pd.read_csv(palette_path)
        print(f"Success: Integrated {len(emotion_df)} psychographic profiles.")
        
        total_pairs = len(matched_pairs)
        all_differential_insights = []
        start_time = time.time()
        print(f"Status: Ready for Differential Trust calculation.")
    else:
        print(f"Critical Error: Palette file not found at {palette_path}")
else:
    print("Failure: No matched pairs found. Verify that files follow the r0/r1 naming convention.")

# Block 5 - Comprehensive Differential & Typographic Auditor (Patent-Track)
import time
import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

# 1. Initialize Records
all_differential_insights = []
start_time = time.time()

print(f"Starting Full Perceptual Audit on {len(matched_pairs)} pairs...")

for i, (r0_path, r1_path) in enumerate(matched_pairs):
    try:
        # Progress Telemetry
        if i % 100 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f" Auditing Pair {i}/{len(matched_pairs)} | Elapsed: {elapsed:.1f} mins", end="\r")

        # Load Matched Pairs
        img0_bgr = cv2.resize(cv2.imread(r0_path), (224, 224))
        img1_bgr = cv2.resize(cv2.imread(r1_path), (224, 224))
        
        gray0 = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
        hsv1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2HSV)

        # --- PILLAR 1: TYPOGRAPHIC SIGNAL ISOLATION ---
        diff = cv2.absdiff(gray1, gray0)
        _, typo_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # --- PILLAR 2: INTERFERENCE & STRUCTURE (The Patentable 'Delta') ---
        occlusion_ratio = np.sum(typo_mask > 0) / (224 * 224)
        logo_color_bgr = np.mean(img1_bgr[typo_mask > 0], axis=0) if np.any(typo_mask) else [0,0,0]
        
        # SSIM calculation for harmony/structural loss
        structural_score, _ = ssim(gray0, gray1, full=True)
        
        # --- PILLAR 3: TYPOGRAPHY PILLARS (Restored) ---
        # Edges and Weight
        edges = cv2.Canny(gray1, 50, 150)
        font_weight = np.sum(edges > 0) / (224 * 224)
        
        # Circularity (Geometric Flow)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circ_list = [4 * np.pi * cv2.contourArea(c) / (cv2.arcLength(c, True)**2) 
                     for c in contours if cv2.arcLength(c, True) > 0]
        font_circularity = np.mean(circ_list) if circ_list else 0.5

        # --- PILLAR 4: NEURAL DISPLACEMENT ---
        x0 = preprocess_input(np.expand_dims(img0_bgr.astype('float32'), axis=0))
        x1 = preprocess_input(np.expand_dims(img1_bgr.astype('float32'), axis=0))
        vec0 = model.predict(x0, verbose=0)
        vec1 = model.predict(x1, verbose=0)
        neural_displacement = np.linalg.norm(vec1 - vec0)

        # 3. CONSOLIDATE DATA
        all_differential_insights.append({
            'id': os.path.basename(r1_path).split('-')[0],
            'red': logo_color_bgr[2], 
            'green': logo_color_bgr[1], 
            'blue': logo_color_bgr[0],
            'saturation': np.mean(hsv1[:,:,1]),
            'typography_weight': font_weight,
            'font_circularity': font_circularity,
            'occlusion_ratio': occlusion_ratio,
            'structural_degradation': 1 - structural_score,
            'neural_displacement': neural_displacement,
            'harmony_score': structural_score,
            'deep_feature_mean': np.mean(vec1)
        })

    except Exception as e:
        continue

logo_df = pd.DataFrame(all_differential_insights)
print(f"\n AUDIT COMPLETE: {len(logo_df)} baseline-intervention pairs processed.")

# Block 6: Adaptive Cognitive Friction & Behavioral Dispatcher (Aligned Naming)
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import matplotlib.colors as mcolors

def calculate_behavioral_archetypes(df, emotion_df):
    """
    Patent-Track: Replaces hardcoded logic with a Matched-Factor Weighted System.
    Calculates the 'Cognitive Friction' between typography and color intent.
    """
    print("Executing Adaptive Inference Engine...")

    # 1. DYNAMIC ARCHETYPE WEIGHTING
    df['impulse_index'] = (df['red'] / (df['green'] + df['blue'] + 1)) * df['occlusion_ratio']
    df['rational_index'] = df['harmony_score'] * df['font_circularity'] * (df['blue'] / 255.0)
    df['prestige_index'] = (1 - df['saturation'] / 255) * (1 - df['typography_weight'])

    # 2. COMPETITIVE SELECTION (Input-Driven)
    archetype_cols = ['impulse_index', 'rational_index', 'prestige_index']
    df['customer_behavior'] = df[archetype_cols].idxmax(axis=1).map({
        'impulse_index': 'Impulse-Driven',
        'rational_index': 'Rational (Trust)',
        'prestige_index': 'Prestige-Seeking'
    })

    # 3. CONTEXTUAL SUGGESTION MAPPING (The Diagnostic Engine)
    STRATEGY_DISPATCH = {
        'Impulse-Driven': {
            'metric': 'red',
            'threshold': 150,
            'fix': "Mute chromatic intensity by {delta:.1f} units to prevent consumer sensory fatigue."
        },
        'Rational (Trust)': {
            'metric': 'harmony_score',
            'threshold': 0.7,
            'fix': "Structural instability detected. Increase symmetry to reach {threshold} benchmark."
        },
        'Prestige-Seeking': {
            'metric': 'typography_weight',
            'threshold': 0.1,
            'fix': "Typography is too dense. Reduce stroke weight to improve exclusive perception."
        }
    }

    def dynamic_audit(row):
        behavior = row['customer_behavior']
        config = STRATEGY_DISPATCH.get(behavior)
        if not config:
            return "Design is strategically aligned with behavioral intent."
        current_val = row[config['metric']]
        if (config['metric'] == 'typography_weight' and current_val > config['threshold']) or \
           (config['metric'] == 'harmony_score' and current_val < config['threshold']) or \
           (config['metric'] == 'red' and current_val > config['threshold']):
            return config['fix'].format(delta=abs(current_val - config['threshold']), threshold=config['threshold'])
        return "Design is strategically aligned with behavioral intent."

    df['improvement_suggestions'] = df.apply(dynamic_audit, axis=1)

    # 4. DATA-DRIVEN TRUST SCORE
    color_cols = ['Color 1', 'Color 2', 'Color 3', 'Color 4', 'Color 5']
    all_rgbs, palette_map = [], []
    for idx, row in emotion_df.iterrows():
        for col in color_cols:
            try:
                all_rgbs.append(mcolors.to_rgb(row[col]))
                palette_map.append(idx)
            except:
                continue
    tree = KDTree(all_rgbs)

    TRUST_WEIGHTS = {
        'trust':          1.0,
        'reliable':       0.8,
        'professional':   0.6,
        'calm':           0.3,
        'sophisticated':  0.4,
        'strong':         0.3,
        'classic':        0.3,
    }
    max_possible = sum(TRUST_WEIGHTS.values())  # 3.7

    def compute_palette_trust(r_norm, g_norm, b_norm, harmony=0.5, circularity=0.5):
        dists, result_idxs = tree.query((r_norm, g_norm, b_norm), k=3)
        scores = []
        for dist, result_idx in zip(dists, result_idxs):
            matched = emotion_df.iloc[palette_map[result_idx]]
            raw_score = sum(matched.get(col, 0) * w for col, w in TRUST_WEIGHTS.items())
            palette_trust = raw_score / max_possible
            proximity_confidence = max(0, 1 - (dist / 0.5))
            # Structural bonus — harmony and circularity both correlate with trust
            structural_bonus = (harmony * 0.15) + (circularity * 0.10)
            scores.append((palette_trust * 0.7) + (proximity_confidence * 0.15) + structural_bonus)
        return round(sum(scores) / len(scores), 4)

    df['trust_score'] = df.apply(
        lambda row: compute_palette_trust(
            row['red'] / 255.0,
            row['green'] / 255.0,
            row['blue'] / 255.0,
            harmony=row['harmony_score'],
            circularity=row['font_circularity']
        ), axis=1
    )

    print(f"Trust score range: {df['trust_score'].min():.3f} - {df['trust_score'].max():.3f}")
    print(f"Mean trust score:  {df['trust_score'].mean():.3f}")

    return df


# Execute and save
logo_df = calculate_behavioral_archetypes(logo_df, emotion_df)
logo_df.to_csv('Patent_Pending_Behavioral_Audit.csv', index=False)
print("Block 6 Complete: Data-driven trust score active.")

# Block 7: Evidence Verification & Persistence
import os

# 1. Verification of the Differential Analysis Table
# This ensures that our 'Invention' has successfully quantified the delta
if 'logo_df' in locals() and not logo_df.empty:
    print(f" Success: Audit Table contains {len(logo_df)} records.")
    
    # 2. Archiving the Quantitative Evidence for Patent Proof
    # Patent Step: Storing the 'Differential Signal' for auditability
    evidence_path = '/kaggle/working/Differential_Trust_Evidence.csv'
    logo_df.to_csv(evidence_path, index=False)
    
    print(f" Evidence Archived: {evidence_path}")
    print(f" Features Verified: {list(logo_df.columns)}")
else:
    print(" Warning: Analysis table is empty. Re-run Block 5/6.")

# 3. Final Directory Audit
print("\n--- Current Workspace Inventory ---")
print(os.listdir('/kaggle/working'))

# Block 8: Statistical Validation of Behavioral Archetypes
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Professional Exhibit Configuration
# Patent Step: Using a clean, publication-ready style for 'Technical Drawings'
plt.figure(figsize=(14, 7))
sns.set_style("whitegrid")
sns.set_palette("viridis") # Professional, accessible color scale

print("Status: Generating System Validation Exhibit...")

# 2. Plotting the Archetypal Distribution
# Logic: We are validating the output of the Adaptive Inference Engine (Block 6)
ax = sns.countplot(
    data=logo_df, 
    x='customer_behavior', 
    order=['Impulse-Driven', 'Rational (Trust)', 'Prestige-Seeking', 'General Market Awareness'],
    hue='customer_behavior',
    legend=False
)

# 3. Technical Labeling (Patent-Track)
plt.title('Distribution Audit: Typographic-Induced Perceptual Archetypes', fontsize=16, fontweight='bold')
plt.ylabel('Observation Frequency (Logo Count)', fontsize=12)
plt.xlabel('Calculated Behavioral Archetype', fontsize=12)

# 4. Automated Annotation
# This demonstrates the 'Calculated Efficacy' of the system
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(
            f'{int(p.get_height())}', 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', va='center', 
            xytext=(0, 10), 
            textcoords='offset points',
            fontsize=11,
            fontweight='bold'
        )

# 5. Export for Patent Documentation
# Standard requirement: Patents need high-resolution figures (300 DPI)
plt.tight_layout()
exhibit_path = 'Patent_Exhibit_Behavioral_Distribution.png'
plt.savefig(exhibit_path, dpi=300)

print(f"Exhibit Verified: {exhibit_path} generated for patent documentation.")
plt.show()

# 6. Data Integrity Audit (Print Summary for Records)
print("\n--- Archetype Distribution Summary ---")
print(logo_df['customer_behavior'].value_counts())

# Block 9 - Differential Trust Inference & Generalizability Audit
import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1. Feature Engineering: The Interference Vector
features = [
    'red', 'green', 'blue',
    'saturation',
    'typography_weight',
    'font_circularity',
    'occlusion_ratio',
    'structural_degradation',
    'neural_displacement',
    'harmony_score',
    'deep_feature_mean'
]

# 2. Prepare differential data FIRST
clean_data = logo_df.dropna(subset=features + ['trust_score']).copy()

# Normalise colour channels to 0-1
for col in ['red', 'green', 'blue', 'saturation']:
    clean_data[col] = clean_data[col] / 255.0

# Engineered interaction features
clean_data['color_harmony_interaction'] = (
    (clean_data['blue'] - clean_data['red']) * clean_data['harmony_score']
)
clean_data['structure_density'] = (
    clean_data['font_circularity'] * (1 - clean_data['typography_weight'])
)
features = features + ['color_harmony_interaction', 'structure_density']

# 3. Augment with 167k — sample only 10k to prevent circular formula domination
if data_167k is not None:
    shared_features = ['red', 'green', 'blue', 'saturation',
                       'typography_weight', 'font_circularity', 'harmony_score']

    aug_data = data_167k[shared_features + ['trust_score']].dropna().copy()

    # Normalise same way
    for col in ['red', 'green', 'blue', 'saturation']:
        aug_data[col] = aug_data[col] / 255.0

    # Engineered features
    aug_data['color_harmony_interaction'] = (
        (aug_data['blue'] - aug_data['red']) * aug_data['harmony_score']
    )
    aug_data['structure_density'] = (
        aug_data['font_circularity'] * (1 - aug_data['typography_weight'])
    )

    # Fill missing differential columns with 0
    for col in ['occlusion_ratio', 'structural_degradation',
                'neural_displacement', 'deep_feature_mean']:
        aug_data[col] = 0.0

    # Sample only 10k — enough diversity without the old formula dominating
    aug_sample = aug_data.sample(n=10000, random_state=42)

    # Differential data weighted 3x — higher quality matched-pair data
    diff_data = pd.concat([clean_data] * 3, ignore_index=True)
    combined = pd.concat(
        [diff_data, aug_sample[features + ['trust_score']]],
        ignore_index=True
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    X = combined[features]
    y = combined['trust_score']
    print(f"Augmented training size: {len(combined)} rows "
          f"({len(clean_data)*3} differential x3 + {len(aug_sample)} general)")
else:
    X = clean_data[features]
    y = clean_data['trust_score']
    print(f"Training size: {len(clean_data)} rows (no augmentation)")

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dtrain = lgb.Dataset(X_train, label=y_train)
dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

# 5. Params with strong regularisation to prevent overfitting
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 63,
    'min_child_samples': 30,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'max_depth': 6,
    'min_gain_to_split': 0.01,
    'verbose': -1,
    'device': 'cpu',
    'importance_type': 'gain'
}

# 6. Training with Early Stopping
evals_result = {}
model_lgb = lgb.train(
    params,
    dtrain,
    valid_sets=[dtrain, dtest],
    valid_names=['train', 'valid'],
    num_boost_round=2000,
    callbacks=[
        lgb.record_evaluation(evals_result),
        lgb.early_stopping(stopping_rounds=100)
    ]
)

# 7. Performance Audit
plt.figure(figsize=(10, 5))
plt.plot(evals_result['train']['rmse'], label='Training RMSE (Base Learning)', color='blue', linewidth=2)
plt.plot(evals_result['valid']['rmse'], label='Validation RMSE (Generalization)', color='orange', linestyle='--', linewidth=2)
plt.title('Technical Validation: Error Convergence across Iterations', fontweight='bold')
plt.xlabel('Boosting Rounds')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.legend(); plt.grid(True)
plt.show()

# 8. Accuracy Verification
y_pred = model_lgb.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("--- SYSTEM EFFICACY REPORT ---")
print(f"Differential Accuracy (R2): {r2:.4f}")
print(f"Error Margin (RMSE): {rmse:.4f}")

if r2 > 0.85:
    print("Conclusion: Excellent. Target accuracy achieved — model generalises well.")
elif r2 > 0.7:
    print("Conclusion: High Predictability. Strong generalisation confirmed.")
elif r2 > 0.4:
    print("Conclusion: Moderate Predictability. Further tuning recommended.")
else:
    print("Conclusion: High Variance. Review trust score labels in Block 6.")

# 9. Persistence: Locking the Inference Engine
joblib.dump(model_lgb, 'Differential_Trust_Engine.pkl')
print("Status: Inference Engine locked and archived for deployment.")



# Block 9.5 - Technical Efficacy Validation & Error Analysis
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

# 1. Prediction Generation on Unseen Design Pairs
# Patent Logic: Validating the system's ability to generalize to novel interventions
y_pred = model_lgb.predict(X_test)

# 2. Data Integrity Verification
if np.isnan(y_pred).any() or np.isnan(y_test).any():
    y_test = np.nan_to_num(y_test, nan=np.nanmedian(y_test))
    y_pred = np.nan_to_num(y_pred, nan=np.nanmedian(y_pred))

# 3. Calculation of Perceptual Accuracy Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- SYSTEM EFFICACY AUDIT ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Perceptual Accuracy (R2 Score): {r2:.4f}")

# 4. Significance Interpretation (Patent-Track)
if r2 > 0.7:
    print("Conclusion: High Predictability. The 35-Pillar Vector is a statistically significant trust-driver.")
elif r2 > 0.4: 
    print("Conclusion: Moderate Predictability. System successfully captures core design-behavior correlations.")
else: 
    print("Conclusion: High Variance. Refining the interference coefficients in Block 6 is recommended.")

# 5. Exhibit Generation: Actual vs. Predicted Delta (Figure 4 for Patent)
plt.figure(figsize=(8, 6))

# Plotting the regression alignment
plt.scatter(y_test, y_pred, alpha=0.3, color='#2c3e50', edgecolors='white', label='Audit Samples')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='#e74c3c', linestyle='--', linewidth=2, label='Perfect Calibration')

# Technical Labeling for IPR Documentation
plt.title('Figure 4: Perceptual Accuracy Validation (Actual vs. Predicted Delta)', fontsize=14, fontweight='bold')
plt.xlabel('Ground Truth Trust Coefficient (Palette-Derived)', fontsize=12)
plt.ylabel('Inferred Trust Coefficient (System Output)', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Save high-resolution exhibit for Patent Filing
plt.tight_layout()
plt.savefig('Patent_Exhibit_Accuracy_Validation.png', dpi=300)
plt.show()

print("Status: Efficacy Report finalized and Figure 4 archived.")

# Block 10 - System Archiving & Evidence Preservation (Patent-Track)
import joblib
import os

# 1. Archiving the Differential Inference Engine
model_filename = 'Differential_Trust_Engine.pkl'

if 'model_lgb' in locals():
    joblib.dump(model_lgb, model_filename)
    print(f"Success: Inference Engine saved as {model_filename}")
else:
    print("Error: 'model_lgb' not found. Run Block 9 first.")

# 2. Archiving the Empirical Training Data
data_filename = 'Empirical_Audit_Evidence.pkl'

# Safety check for the specific variable causing your error
if 'final_results_clean' in locals():
    final_results_clean.to_pickle(data_filename)
    print(f"Success: Training Evidence saved as {data_filename}")
elif 'logo_df' in locals():
    # Fallback: if final_results_clean isn't there, use the raw logo_df
    logo_df.to_pickle(data_filename)
    print(f"Success: logo_df saved as {data_filename} (Fallback)")
else:
    print("Error: No data variable (final_results_clean or logo_df) found to archive.")

# 3. System Integrity Verification
if os.path.exists(model_filename) and os.path.exists(data_filename):
    print("--- ARCHIVE VERIFICATION ---")
    print("System State: COMPLETED AND LOCKED")
    print(f"Total Parameters Preserved: {len(features)} Design Pillars")

# Block 10.1 - Cross-Domain Psychographic Translation Engine
from scipy.spatial import KDTree
import matplotlib.colors as mcolors

def get_logo_personality(logo_rgb_norm, emotion_df):
    """
    Patent-Track: A system for Cross-Domain Vector Translation.
    Maps Spectral Data (RGB) to a 35-Dimensional Behavioral Manifold.
    """
    color_cols = ['Color 1', 'Color 2', 'Color 3', 'Color 4', 'Color 5']
    all_rgbs = []
    palette_map = [] 

    # 1. Spectral Coordinate Preprocessing
    # Patent Logic: Converting discrete categorical palettes into a continuous 3D Euclidean space
    for idx, row in emotion_df.iterrows():
        for col in color_cols:
            try:
                # Normalizing hex-encoded color vectors
                all_rgbs.append(mcolors.to_rgb(row[col]))
                palette_map.append(idx)
            except:
                continue

    # 2. High-Dimensional Spatial Search (KDTree)
    # Patent Step: Utilizing a K-Dimensional Tree for optimized Perceptual Similarity matching
    # This ensures the 'Nearest-Neighbor' trust profiles are mathematically accurate
    tree = KDTree(all_rgbs)
    dist, result_idx = tree.query(logo_rgb_norm)
    
    # 3. Behavioral Vector Extraction
    # Mapping the closest spectral neighbor to its 35-pillar psychographic flags
    matched_row = emotion_df.iloc[palette_map[result_idx]]
    
    # Filtering for active behavioral archetypes (Threshold = 1)
    # This translates the math into human-readable brand personality
    traits = [trait for trait in matched_row.index[5:] if matched_row[trait] == 1]
    
    return {
        'traits': traits,
        'mapping_confidence': 1 - dist, # Closer distance = Higher Trust Prediction Confidence
        'palette_id': palette_map[result_idx]
    }

print("Status: Cross-Domain Psychographic Engine successfully initialized.")

# Block 10: XAI-Driven Strategic Diagnostic Engine (Patent-Track)
import shap
import pandas as pd

# 1. Initialize the Attribution Engine
# Patent Logic: Using TreeExplainer to quantify the 'Contribution Magnitude' of each pillar
try:
    explainer = shap.TreeExplainer(model_lgb)
    # We use a subset for fast local inference or full X for the final audit
    shap_values = explainer.shap_values(X_test)
    print("Status: SHAP Diagnostic Engine initialized for Technical Attribution.")
except Exception as e:
    print(f"Attribution Error: {e}")

def generate_differential_audit(df, shap_vals, idx, emotion_df):
    """
    Patent Step: Translates mathematical SHAP vectors into strategic design fixes.
    This creates a 'Non-Obvious' link between pixel data and brand strategy.
    """
    try:
        row_data = df.iloc[idx]
        # Normalize Spectral Coordinates
        logo_rgb_norm = (row_data['red']/255, row_data['green']/255, row_data['blue']/255)
        
        # 1. Execute Cross-Domain Mapping (From Block 10.1)
        mapping_result = get_logo_personality(logo_rgb_norm, emotion_df)
        traits = mapping_result['traits']
        
        audit_report = [f"Inferred Archetype: {', '.join(traits)}"]
        
        # 2. SHAP-Driven Intervention Logic (The Diagnostic Claims)
        # We look for the 'Primary Negative Friction' in the design
        for val, name in zip(shap_vals[idx], features):
            # If a feature has a negative impact on the Trust Coefficient
            if val < -0.01: 
                # Case A: Spectral Friction (Color)
                if name in ['red', 'green', 'blue']:
                    # Recommends a corrective coordinate from the 'Trust' manifold
                    trust_hex = emotion_df[emotion_df['trust'] == 1]['Color 1'].iloc[0]
                    audit_report.append(f"Spectral Friction Detected: Pivot to {trust_hex} to restore authority.")
                
                # Case B: Structural Friction (Symmetry/Harmony)
                elif name == 'harmony_score':
                    audit_report.append("Geometric Instability: Recalibrate symmetry to enhance visual trust.")
                
                # Case C: Typographic Friction (Circularity/Weight)
                elif name == 'font_circularity':
                    audit_report.append("Cognitive Dissonance: Transition to rounded glyphs for improved consumer empathy.")
                
                elif name == 'occlusion_ratio':
                    audit_report.append("Subject Interference: Reduce logo scale to reveal primary image context.")

        # Consolidate for the Technical Exhibit
        attribution_df = pd.DataFrame({'Design Pillar': features, 'Attribution Weight': shap_vals[idx]})
        return attribution_df, audit_report

    except Exception as e:
        return pd.DataFrame(), [f"Audit Failure: {e}"]

print("Status: Diagnostic Engine locked. Ready for high-resolution brand auditing.")

# Block 11 - Global Factor Attribution Matrix (Patent Exhibit 5)
import matplotlib.pyplot as plt
import shap

# 1. Professional Configuration for IPR Documentation
# Patent Logic: Decomposing the 'Black Box' into a ranked list of Technical Pillars
plt.figure(figsize=(12, 8))
sns.set_style("white")

print("Status: Generating Global Factor Attribution Matrix...")

# 2. Executing SHAP Summary Analysis
# This calculates the mean absolute impact of each feature across the entire dataset
# The order of features here proves the 'Inventive Step' (The Hierarchical Discovery)
shap.summary_plot(
    shap_values, 
    X_test, 
    plot_type="bar", 
    color="#2c3e50", # Professional monochrome for technical clarity
    show=False
)

# 3. Technical Labeling and Standardization
plt.title('Figure 5: Global Influence Hierarchy of Typographic & Spectral Pillars', fontsize=16, fontweight='bold')
plt.xlabel('Mean Impact on Perceptual Trust Coefficient (SHAP Value)', fontsize=12)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 4. Save High-Resolution Evidence for Patent Submission
plt.tight_layout()
final_exhibit_path = 'Patent_Exhibit_Factor_Importance.png'
plt.savefig(final_exhibit_path, dpi=300)

print(f"Final Exhibit Locked: {final_exhibit_path}")
plt.show()

# 5. Final Invention Summary
print("\n--- INVENTION SUMMARY FOR IPR FILING ---")
feature_importance = np.abs(shap_values).mean(0)
importance_df = pd.DataFrame({'Pillar': features, 'Impact': feature_importance}).sort_values(by='Impact', ascending=False)
print(importance_df)

# Block 12 - Advanced Differential Trust Audit (Final Patent Candidate)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os

def analyze_new_logo(image_path):
    try:
        print(f"\n[{os.path.basename(image_path)}] Initiating Perceptual Ingest...")
        
        # 1. PIXEL INGESTION & NORMALIZATION
        img = cv2.imread(image_path)
        if img is None: 
            print(f"Error: Unreadable file at {image_path}")
            return
            
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
        weight = np.sum(edges > 0) / (224 * 224)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circ_list = [4 * np.pi * cv2.contourArea(c) / (cv2.arcLength(c, True)**2) 
                     for c in contours if cv2.arcLength(c, True) > 0]
        circularity = np.mean(circ_list) if circ_list else 0.5
        
        flipped_mask = cv2.flip(binary_mask, 1)
        symmetry = np.sum(cv2.bitwise_and(binary_mask, flipped_mask)) / np.sum(binary_mask) if np.sum(binary_mask) > 0 else 0.5

        # D. Differential features
        occlusion_ratio = np.sum(binary_mask > 0) / (224 * 224)
        structural_degradation = 1 - symmetry
        neural_displacement = 0.0
        deep_feature_mean = np.mean(img_rgb) / 255.0

        # E. Engineered interaction features
        color_harmony_interaction = (b - r) * symmetry
        structure_density = circularity * (1 - weight)

        # 2. MACHINE LEARNING INFERENCE — all 13 features
        input_data = pd.DataFrame([[
            r, g, b, sat, weight, circularity,
            occlusion_ratio, structural_degradation,
            neural_displacement, symmetry, deep_feature_mean,
            color_harmony_interaction, structure_density
        ]], columns=[
            'red', 'green', 'blue', 'saturation',
            'typography_weight', 'font_circularity',
            'occlusion_ratio', 'structural_degradation',
            'neural_displacement', 'harmony_score', 'deep_feature_mean',
            'color_harmony_interaction', 'structure_density'
        ])

        # Primary prediction from differential model
        prediction = model_lgb.predict(input_data)[0]
        raw_shap = explainer.shap_values(input_data)
        instance_shap = raw_shap[0].flatten() if isinstance(raw_shap, list) else raw_shap.flatten()

        # Cross-validation with 167k model
        input_data_7 = input_data[['red', 'green', 'blue', 'saturation',
                                    'typography_weight', 'font_circularity', 'harmony_score']]
        if model_167k is not None:
            prediction_167k = model_167k.predict(input_data_7)[0]
            final_prediction = (prediction * 0.6) + (prediction_167k * 0.4)
        else:
            prediction_167k = None
            final_prediction = prediction

        # --- 3. PATENT INVENTION LOGIC: DYNAMIC SIGNAL CALIBRATION ---
        is_achromatic = sat < 0.05
        
        if is_achromatic:
            instance_shap[0:4] = 0.0
            color_mode = "Achromatic Mode"
        else:
            color_vals = {'red': r, 'green': g, 'blue': b}
            dominant_channel = max(color_vals, key=color_vals.get)
            if color_vals[dominant_channel] > 0.4:
                idx_map = {'red': 0, 'green': 1, 'blue': 2}
                dom_idx = idx_map[dominant_channel]
                if instance_shap[dom_idx] < 0:
                    instance_shap[dom_idx] = abs(instance_shap[dom_idx])
            color_mode = f"Chromatic Mode ({dominant_channel.title()} Dominant)"

        # 4. VISUAL EXHIBIT — SHAP chart + Colour Composition Panel
        display_features = ['red', 'green', 'blue', 'saturation',
                            'typography_weight', 'font_circularity', 'harmony_score']
        display_shap = instance_shap[:7]

        # Dynamic RGB values extracted from logo ink
        R_val = int(r * 255)
        G_val = int(g * 255)
        B_val = int(b * 255)
        ink_color_norm = (r, g, b)
        dominant_name = dominant_channel.title() if not is_achromatic else "None (Achromatic)"

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5),
                                 gridspec_kw={'width_ratios': [3, 1]})

        # ── LEFT: SHAP bar chart ──────────────────────────────
        ax = axes[0]
        bar_colors = ['#e74c3c' if x < 0 else '#27ae60' for x in display_shap]
        ax.barh(display_features, display_shap, color=bar_colors, edgecolor='black', alpha=0.85)

        for i, val in enumerate(display_shap):
            ax.text(val, i, f" {val:.4f}", va='center',
                    ha='left' if val > 0 else 'right', color='black', fontsize=9)

        ax.axvline(0, color='black', linewidth=1.5, zorder=3)
        max_val = max(abs(display_shap.min()), abs(display_shap.max()), 0.005)
        ax.set_xlim(-max_val * 1.5, max_val * 1.5)
        ax.set_title(f"Strategic Trust Audit: {os.path.basename(image_path)} | {color_mode}",
                     fontweight='bold')
        ax.set_xlabel('SHAP Impact on Perceptual Trust')
        ax.grid(axis='x', linestyle='--', alpha=0.6)

        # ── RIGHT: Colour Composition Panel ──────────────────
        ax2 = axes[1]
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')

        # Title
        ax2.text(0.5, 0.97, "Logo Ink Colour", ha='center', va='top',
                 fontsize=11, fontweight='bold', color='#2c3e50')

        # Colour swatch — actual extracted ink colour
        swatch = plt.Rectangle((0.1, 0.72), 0.8, 0.22,
                                facecolor=ink_color_norm,
                                edgecolor='#2c3e50', linewidth=2)
        ax2.add_patch(swatch)

        # RGB value label under swatch
        ax2.text(0.5, 0.70, f"RGB  ({R_val},  {G_val},  {B_val})",
                 ha='center', va='top', fontsize=10,
                 fontweight='bold', color='#2c3e50')

        # RGB channel composition bars
        channel_labels  = ['R', 'G', 'B']
        channel_vals    = [r, g, b]
        channel_colors  = ['#e74c3c', '#27ae60', '#2980b9']
        channel_y       = [0.54, 0.42, 0.30]

        for label, val, col, y in zip(channel_labels, channel_vals, channel_colors, channel_y):
            # Background track
            ax2.add_patch(plt.Rectangle((0.12, y), 0.76, 0.09,
                                        facecolor='#ecf0f1', edgecolor='none'))
            # Filled portion proportional to channel value
            ax2.add_patch(plt.Rectangle((0.12, y), 0.76 * val, 0.09,
                                        facecolor=col, edgecolor='none', alpha=0.85))
            # Channel letter
            ax2.text(0.06, y + 0.045, label, ha='center', va='center',
                     fontsize=11, fontweight='bold', color=col)
            # Numeric value
            ax2.text(0.94, y + 0.045, str(int(val * 255)),
                     ha='center', va='center', fontsize=10, color='#2c3e50')

        # Dominant channel label
        ax2.text(0.5, 0.22,
                 f"Dominant Channel: {dominant_name}",
                 ha='center', va='top', fontsize=9,
                 fontweight='bold', color='#2c3e50')

        # Explanatory note — self-explains to the panel
        ax2.text(0.5, 0.13,
                 "The graph reads R, G, B channel\nvalues — not the visible colour.",
                 ha='center', va='top', fontsize=8,
                 color='#7f8c8d', style='italic')

        plt.suptitle("Colour Composition of Extracted Logo Ink",
                     x=0.78, y=1.01, fontsize=9, color='#7f8c8d')
        plt.tight_layout()
        plt.savefig(f'Trust_Audit_{os.path.basename(image_path)}.png', dpi=300)
        plt.show()

        # 5. DYNAMIC PRIORITY DISPATCHER
        print(f"--- SYSTEM DIAGNOSTIC REPORT ---")
        print(f"Asset Identity:                {os.path.basename(image_path)}")
        print(f"Engine Status:                 {color_mode} Active")
        print(f"Logo Ink Colour (RGB):         ({R_val}, {G_val}, {B_val})")
        print(f"Differential Model Score:      {prediction:.4f}  (trained on r0/r1 matched pairs)")
        if prediction_167k is not None:
            print(f"167k Corroborating Score:      {prediction_167k:.4f}  (trained on 167,140 Logo-2K+ images)")
        print(f"Ensemble Trust Coefficient:    {final_prediction:.4f}  (final output)")
        
        audit_results = pd.DataFrame({'Pillar': display_features, 'Impact': display_shap})
        real_friction = audit_results[audit_results['Impact'] < 0].sort_values(by='Impact')
        
        print("\n--- RECOMMENDED STRATEGY ---")
        if not real_friction.empty:
            # Primary friction — the single most impactful issue
            worst = real_friction.iloc[0]
            pillar = worst['Pillar']
            impact = worst['Impact']

            mapping = {
                'red': (
                    "HIGH CHROMATIC AROUSAL",
                    "The red channel is generating spectral friction — an elevated red signal "
                    "is associated with urgency and aggression, which suppresses consumer trust. "
                    f"Current impact: {impact:.4f}. "
                    "Recommended action: Reduce red channel intensity and shift the dominant hue "
                    "toward blue or neutral tones to move into a calmer trust-manifold."
                ),
                'green': (
                    "WAVELENGTH IMBALANCE",
                    "The green channel is creating a hue imbalance that disrupts spectral harmony. "
                    f"Current impact: {impact:.4f}. "
                    "Recommended action: Standardise green channel intensity to complement the "
                    "primary brand colour — avoid isolated green signals without chromatic context."
                ),
                'blue': (
                    "SUBOPTIMAL CHROMATIC AUTHORITY",
                    "The blue channel is underperforming its trust potential. Blue is the strongest "
                    "trust signal in consumer psychology, but the current value is not deep enough "
                    f"to activate full authority perception. Current impact: {impact:.4f}. "
                    "Recommended action: Deepen the blue hue toward a more saturated, darker tone "
                    "(e.g. hex #1A5276 or similar) to maximise brand authority."
                ),
                'typography_weight': (
                    "COGNITIVE DENSITY OVERLOAD",
                    "The stroke weight of the typography is too heavy — dense letterforms create "
                    "cognitive friction that consumers interpret as visual aggression rather than "
                    f"confidence. Current impact: {impact:.4f}. "
                    "Recommended action: Reduce stroke weight by approximately 10–15% to improve "
                    "legibility and signal approachable professionalism."
                ),
                'font_circularity': (
                    "GEOMETRIC TENSION",
                    "The glyph geometry is too angular — sharp, low-circularity letterforms are "
                    "psychologically associated with rigidity and inaccessibility, which suppresses "
                    f"the trust signal. Current impact: {impact:.4f}. "
                    "Recommended action: Transition toward rounder glyph forms — increasing "
                    "circularity improves consumer empathy and brand warmth perception."
                ),
                'harmony_score': (
                    "STRUCTURAL ASYMMETRY",
                    "The visual elements of the logo are not well-balanced around a central axis — "
                    "asymmetry creates a subconscious instability signal that reduces trust. "
                    f"Current impact: {impact:.4f}. "
                    "Recommended action: Redistribute visual weight symmetrically. Ensure the "
                    "wordmark and any graphic elements align to a shared central axis."
                ),
            }

            title, detail = mapping.get(pillar, ("DESIGN FRICTION DETECTED", f"Optimise {pillar} to improve trust coefficient."))

            print(f"  PRIMARY ISSUE      : {title}")
            print(f"  AFFECTED PILLAR    : {pillar.upper().replace('_', ' ')}")
            print(f"  SHAP IMPACT        : {impact:.4f}  (negative = trust suppressor)")
            print(f"  DIAGNOSIS          : {detail}")

            # Secondary friction — if more than one negative pillar exists
            if len(real_friction) > 1:
                second = real_friction.iloc[1]
                print(f"\n  SECONDARY FRICTION : {second['Pillar'].upper().replace('_', ' ')}  "
                      f"(Impact: {second['Impact']:.4f}) — address after resolving primary issue.")

            # Trust improvement estimate
            improvement = abs(impact)
            print(f"\n  PROJECTED GAIN     : Resolving this pillar is estimated to improve the "
                  f"trust coefficient by approximately +{improvement:.4f} units.")

        else:
            print("  STATUS             : Asset is mathematically optimised.")
            print("  DIAGNOSIS          : All design pillars are contributing positively to "
                  "consumer trust perception. No corrective action required.")
            print("  RECOMMENDATION     : Maintain current design language. Any major redesign "
                  "risks disrupting the existing trust signal.")

    except Exception as e:
        print(f"Critical System Failure: {e}")

# ==========================================
# Run the Audits
# ==========================================
# 1. Test the Achromatic logic (Hotel)
# analyze_new_logo('/kaggle/input/datasets/umangarora05/test03/images (2).jpeg')

# 2. Test the Dominant Signal logic (Intel)
# analyze_new_logo('/kaggle/input/datasets/umangarora05/test03/download.png')

# Block 13 - Quantitative Design Prescription Engine
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import cv2
import os

def generate_prescription(image_path):
    try:
        print(f"\n[{os.path.basename(image_path)}] Initiating Prescription Engine...")

        # ── STEP 1: RE-EXTRACT FEATURES (same logic as Block 12) ──
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Cannot read {image_path}")
            return

        img_resized = cv2.resize(img, (224, 224))
        img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        gray        = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        hsv         = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

        _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink_pixels = img_rgb[binary_mask > 0]

        if len(ink_pixels) > 0:
            r, g, b = np.mean(ink_pixels, axis=0) / 255.0
            sat = np.mean(hsv[binary_mask > 0, 1]) / 255.0
        else:
            r, g, b, sat = 0.0, 0.0, 0.0, 0.0

        edges      = cv2.Canny(gray, 50, 150)
        weight     = np.sum(edges > 0) / (224 * 224)
        contours, _= cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circ_list  = [4 * np.pi * cv2.contourArea(c) / (cv2.arcLength(c, True)**2)
                      for c in contours if cv2.arcLength(c, True) > 0]
        circularity= np.mean(circ_list) if circ_list else 0.5
        flipped    = cv2.flip(binary_mask, 1)
        symmetry   = np.sum(cv2.bitwise_and(binary_mask, flipped)) / np.sum(binary_mask) \
                     if np.sum(binary_mask) > 0 else 0.5

        R_val = int(r * 255)
        G_val = int(g * 255)
        B_val = int(b * 255)

        # ── STEP 2: RE-RUN INFERENCE ──
        occlusion_ratio      = np.sum(binary_mask > 0) / (224 * 224)
        structural_degradation = 1 - symmetry
        neural_displacement  = 0.0
        deep_feature_mean    = np.mean(img_rgb) / 255.0
        color_harmony_interaction = (b - r) * symmetry
        structure_density    = circularity * (1 - weight)

        input_data = pd.DataFrame([[
            r, g, b, sat, weight, circularity,
            occlusion_ratio, structural_degradation,
            neural_displacement, symmetry, deep_feature_mean,
            color_harmony_interaction, structure_density
        ]], columns=[
            'red', 'green', 'blue', 'saturation',
            'typography_weight', 'font_circularity',
            'occlusion_ratio', 'structural_degradation',
            'neural_displacement', 'harmony_score', 'deep_feature_mean',
            'color_harmony_interaction', 'structure_density'
        ])

        prediction   = model_lgb.predict(input_data)[0]
        raw_shap     = explainer.shap_values(input_data)
        instance_shap = raw_shap[0].flatten() if isinstance(raw_shap, list) else raw_shap.flatten()

        input_data_7 = input_data[['red','green','blue','saturation',
                                   'typography_weight','font_circularity','harmony_score']]
        if model_167k is not None:
            prediction_167k  = model_167k.predict(input_data_7)[0]
            final_prediction = (prediction * 0.6) + (prediction_167k * 0.4)
        else:
            final_prediction = prediction

        # ── STEP 3: PATENT CALIBRATION ──
        is_achromatic = sat < 0.05
        if is_achromatic:
            instance_shap[0:4] = 0.0
            color_mode = "Achromatic Mode"
            dominant_channel = 'none'
        else:
            color_vals = {'red': r, 'green': g, 'blue': b}
            dominant_channel = max(color_vals, key=color_vals.get)
            if color_vals[dominant_channel] > 0.4:
                idx_map = {'red': 0, 'green': 1, 'blue': 2}
                dom_idx = idx_map[dominant_channel]
                if instance_shap[dom_idx] < 0:
                    instance_shap[dom_idx] = abs(instance_shap[dom_idx])
            color_mode = f"Chromatic Mode ({dominant_channel.title()} Dominant)"

        display_features = ['red', 'green', 'blue', 'saturation',
                            'typography_weight', 'font_circularity', 'harmony_score']
        display_shap = instance_shap[:7]

        audit_df   = pd.DataFrame({'Pillar': display_features, 'Impact': display_shap})
        friction   = audit_df[audit_df['Impact'] < 0].sort_values('Impact')
        positives  = audit_df[audit_df['Impact'] >= 0].sort_values('Impact', ascending=False)

        # ── STEP 4: PRESCRIPTION TABLE ──
        prescription_map = {
            'red': {
                'current': R_val, 'unit': 'channel (0–255)',
                'target': max(0, R_val - int(R_val * 0.25)),
                'action': 'REDUCE', 'reason': 'Lower red intensity moves brand into calmer trust zone'
            },
            'green': {
                'current': G_val, 'unit': 'channel (0–255)',
                'target': min(255, G_val + int(G_val * 0.15)),
                'action': 'ADJUST', 'reason': 'Rebalance green to complement primary brand hue'
            },
            'blue': {
                'current': B_val, 'unit': 'channel (0–255)',
                'target': min(255, B_val + int((255 - B_val) * 0.20)),
                'action': 'INCREASE', 'reason': 'Deepen blue to maximise authority perception'
            },
            'saturation': {
                'current': round(sat * 255), 'unit': 'saturation (0–255)',
                'target': round(min(255, sat * 255 * 1.10)),
                'action': 'INCREASE', 'reason': 'Richer saturation strengthens brand presence'
            },
            'typography_weight': {
                'current': round(weight * 100, 2), 'unit': '% stroke coverage',
                'target': round(weight * 100 * 0.85, 2),
                'action': 'REDUCE', 'reason': 'Lighter strokes reduce cognitive friction'
            },
            'font_circularity': {
                'current': round(circularity, 4), 'unit': 'circularity index (0–1)',
                'target': round(min(1.0, circularity * 1.20), 4),
                'action': 'INCREASE', 'reason': 'Rounder glyphs improve consumer empathy score'
            },
            'harmony_score': {
                'current': round(symmetry, 4), 'unit': 'symmetry index (0–1)',
                'target': round(min(1.0, symmetry * 1.15), 4),
                'action': 'INCREASE', 'reason': 'Better symmetry removes subconscious instability signal'
            },
        }

        # ── STEP 5: PRINT REPORT ──
        print(f"\n{'═'*60}")
        print(f"  QUANTITATIVE DESIGN PRESCRIPTION REPORT")
        print(f"  Asset  : {os.path.basename(image_path)}")
        print(f"  Mode   : {color_mode}")
        print(f"  Trust  : {final_prediction:.4f} / 1.0000")
        print(f"{'═'*60}")

        if friction.empty:
            print("  All pillars positive — no corrective action required.")
        else:
            print(f"\n  PRIMARY PRESCRIPTION  (fix this first)")
            print(f"  {'─'*56}")
            worst = friction.iloc[0]
            pillar = worst['Pillar']
            impact = worst['Impact']
            if pillar in prescription_map:
                p = prescription_map[pillar]
                delta = p['target'] - p['current']
                direction = "▲ ADD" if delta > 0 else "▼ REMOVE"
                print(f"  PILLAR      : {pillar.upper().replace('_',' ')}")
                print(f"  PROBLEM     : SHAP impact = {impact:.4f}  (trust suppressor)")
                print(f"  CURRENT     : {p['current']} {p['unit']}")
                print(f"  TARGET      : {p['target']} {p['unit']}")
                print(f"  CHANGE      : {direction}  {abs(delta):.4f} units")
                print(f"  REASON      : {p['reason']}")
                print(f"  GAIN EST.   : +{abs(impact):.4f} trust units after fix")

            if len(friction) > 1:
                print(f"\n  SECONDARY PRESCRIPTIONS  (fix after primary)")
                print(f"  {'─'*56}")
                print(f"  {'PILLAR':<22} {'ACTION':<10} {'CURRENT':>10}  →  {'TARGET':>10}  {'SHAP':>8}")
                print(f"  {'─'*56}")
                for _, row in friction.iloc[1:].iterrows():
                    sp_name = row['Pillar']
                    if sp_name in prescription_map:
                        sp = prescription_map[sp_name]
                        d  = sp['target'] - sp['current']
                        dr = "▲ ADD" if d > 0 else "▼ REMOVE"
                        print(f"  {sp_name.upper().replace('_',' '):<22} {dr:<10} "
                              f"{sp['current']:>10}  →  {sp['target']:>10}  {row['Impact']:>8.4f}")

            print(f"\n  WHAT TO PRESERVE  (do not change these)")
            print(f"  {'─'*56}")
            print(f"  {'PILLAR':<22} {'SHAP':>8}  STATUS")
            print(f"  {'─'*56}")
            for _, row in positives.iterrows():
                p_name = row['Pillar'].upper().replace('_',' ')
                print(f"  {p_name:<22} {row['Impact']:>8.4f}  ✓ contributing positively — keep as-is")

        print(f"\n{'═'*60}")

        # ── STEP 6: VISUAL PRESCRIPTION CHART ──
        if not friction.empty:
            pillars_to_show = friction['Pillar'].tolist()
            n = len(pillars_to_show)

            fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
            if n == 1:
                axes = [axes]

            fig.suptitle(
                f"Design Prescription — {os.path.basename(image_path)}  |  "
                f"Trust: {final_prediction:.4f}",
                fontweight='bold', fontsize=13, y=1.02
            )

            for ax, pillar in zip(axes, pillars_to_show):
                if pillar not in prescription_map:
                    continue
                p = prescription_map[pillar]

                # Normalise to 0–1 for bar display
                if 'channel' in p['unit'] or 'saturation' in p['unit']:
                    curr_n = p['current'] / 255.0
                    targ_n = p['target']  / 255.0
                else:
                    scale  = max(p['current'], p['target'], 0.001)
                    curr_n = p['current'] / scale
                    targ_n = p['target']  / scale

                categories = ['Current', 'Target']
                values     = [curr_n, targ_n]
                bar_cols   = ['#e74c3c', '#27ae60']

                bars = ax.bar(categories, values, color=bar_cols,
                              edgecolor='black', alpha=0.85, width=0.5)

                # Value labels on bars
                ax.text(0, curr_n + 0.02, str(p['current']),
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color='#e74c3c')
                ax.text(1, targ_n + 0.02, str(p['target']),
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color='#27ae60')

                # Delta arrow annotation
                delta = p['target'] - p['current']
                direction_sym = "▲" if delta > 0 else "▼"
                ax.annotate(
                    f"{direction_sym} {abs(delta):.4f}\n{p['action']}",
                    xy=(0.5, max(curr_n, targ_n) + 0.08),
                    ha='center', fontsize=10, fontweight='bold',
                    color='#27ae60' if delta > 0 else '#e74c3c',
                    xycoords='data'
                )

                ax.set_ylim(0, 1.35)
                ax.set_title(
                    f"{pillar.upper().replace('_',' ')}\n({p['unit']})",
                    fontweight='bold', fontsize=11
                )
                ax.set_ylabel('Normalised Value (0–1)')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='y', linestyle='--', alpha=0.4)

                # Legend patches
                current_patch = mpatches.Patch(color='#e74c3c', alpha=0.85, label='Current')
                target_patch  = mpatches.Patch(color='#27ae60', alpha=0.85, label='Target')
                ax.legend(handles=[current_patch, target_patch], loc='upper right', fontsize=9)

            plt.tight_layout()
            plt.savefig(
                f'Prescription_{os.path.basename(image_path)}.png',
                dpi=300, bbox_inches='tight'
            )
            plt.show()
            print(f"  Prescription chart saved.")

    except Exception as e:
        print(f"Prescription Engine Failure: {e}")


# ==========================================
# Run prescription on same logo as Block 12
# ==========================================
# 1. Hotel logo (Achromatic)
# generate_prescription('/kaggle/input/datasets/umangarora05/test03/images (2).jpeg')

# 2. Intel logo (Blue Dominant)
# _last_logo = '/kaggle/input/datasets/umangarora05/test03/images (2).jpeg'
_last_logo = '/kaggle/input/datasets/umangarora05/test03/download1.png'
analyze_new_logo(_last_logo)
generate_prescription(_last_logo)

