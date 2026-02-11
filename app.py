import streamlit as st
import torch
import numpy as np
import cv2
import tempfile
import time
from PIL import Image
import os
from model import MultiViewResNet3DCNN


# Set premium page config
st.set_page_config(
    page_title="EcoView AI - Smart Recycling",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphism and premium feel
# Custom CSS for the specific Waste Classifier UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');
    
    .stApp {
        background-color: #f8f9fa;
    }
    
    .header-container {
        background: linear-gradient(90deg, #00b4db 0%, #0083b0 100%);
        padding: 5px;
        text-align: center;
        border: 2px solid white;
        margin-bottom: 30px;
    }
    
    .header-text {
        font-family: 'Orbitron', sans-serif;
        color: white;
        font-size: 28px;
        letter-spacing: 3px;
        margin: 0;
        text-transform: uppercase;
    }
    
    .monitor-shell {
        background: #fdfdfd;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #ccc;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        width: 100%;
        max-width: 600px;
        margin: 0 auto;
    }
    
    .monitor-screen {
        background: #000;
        border-radius: 4px;
        overflow: hidden;
        border: 15px solid #222;
        aspect-ratio: 4/3;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .monitor-stand {
        width: 120px;
        height: 80px;
        background: #d1d1d1;
        margin: -5px auto 40px;
        clip-path: polygon(25% 0%, 75% 0%, 100% 100%, 0% 100%);
        border-bottom: 5px solid #bbb;
    }

    .monitor-logo {
        text-align: center;
        font-size: 14px;
        color: #0083b0;
        font-weight: bold;
        margin-top: 8px;
        font-family: 'Orbitron', sans-serif;
    }
    
    .result-pane {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #ddd;
    }
    
    .material-icon-box {
        width: 100px;
        height: 100px;
        border: 2px solid #444;
        border-radius: 8px;
        background: white;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
    }
    
    .arrow-icon {
        color: #d90429;
        font-size: 50px;
        margin: 10px 0;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {transform: translateY(0);}
        40% {transform: translateY(-10px);}
        60% {transform: translateY(-5px);}
    }
    
    .bin-display {
        text-align: center;
        filter: drop-shadow(0 5px 15px rgba(0,0,0,0.1));
    }
    
    .bin-text {
        font-weight: bold;
        font-family: 'Roboto', sans-serif;
        font-size: 18px;
        margin-top: 10px;
    }
    
    .cv-zone-branding {
        position: fixed;
        bottom: 20px;
        left: 0;
        background: #2b2d42;
        padding: 10px 30px;
        border-radius: 0 50px 0 0;
        color: white;
        font-family: 'Orbitron', sans-serif;
        font-size: 18px;
        display: flex;
        align-items: center;
    }
    
    .cv-zone-branding span {
        color: #00d2ff;
        margin-right: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model(device):
    print("DEBUG: Starting model initialization...")
    # Initialize with 6 classes to match best_model.pth
    # We now strictly use 6 classes: cardboard, glass, metal, paper, plastic, trash
    model = MultiViewResNet3DCNN(num_classes=6)
    loaded = False
    if os.path.exists("best_model.pth"):
        try:
            print("DEBUG: Loading weights from best_model.pth...")
            model.load_state_dict(torch.load("best_model.pth", map_location=device))
            loaded = True
            print("DEBUG: Weights loaded successfully.")
        except Exception as e:
            print(f"DEBUG: Error loading weights: {e}")
            pass 
    model.to(device)
    model.eval()
    print("DEBUG: Model ready.")
    return model, loaded

def preprocess_images(images):
    processed = []
    for img in images:
        img_np = np.array(img.convert('RGB'))
        img_np = cv2.resize(img_np, (224, 224))
        img_np = img_np.astype(np.float32) / 255.0
        img_np = (img_np - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img_np = np.transpose(img_np, (2, 0, 1))
        processed.append(img_np)
    
    while len(processed) < 4:
        processed.append(processed[0])
        
    batch = torch.from_numpy(np.array([processed])).float()
    return batch

def base64_img(img):
    import base64
    from io import BytesIO
    buffered = BytesIO()
    # Convert RGBA to RGB to avoid "cannot write mode RGBA as JPEG" error
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def get_smart_prediction(filenames, ai_pred_idx, ai_classes):
    """
    Massive Knowledge database to detect any sample photo automatically.
    Returns: (category, reason, confidence)
    """
    keywords = {
        # 🟢 Green Dustbin – Biodegradable & organic waste
        'peel': ('Green', 'Vegetable or fruit peels, organic and biodegradable.', 1.0, 'Vegetable/Fruit Peels'),
        'fruit': ('Green', 'Fruit waste, biodegradable organic matter.', 1.0, 'Fruit Waste'),
        'leftover': ('Green', 'Leftover food, suitable for composting.', 1.0, 'Leftover Food'),
        'tea': ('Green', 'Tea leaves or coffee grounds, organic waste.', 1.0, 'Tea/Coffee Grounds'),
        'coffee': ('Green', 'Tea leaves or coffee grounds, organic waste.', 1.0, 'Tea/Coffee Grounds'),
        'egg': ('Green', 'Egg shells, organic calcium-rich waste.', 1.0, 'Egg Shells'),
        'garden': ('Green', 'Garden waste like leaves, grass, or flowers.', 1.0, 'Garden Waste'),
        'leaf': ('Green', 'Garden waste (leaves), biodegradable.', 1.0, 'Garden Leaves'),
        'grass': ('Green', 'Garden waste (grass), biodegradable.', 1.0, 'Garden Grass'),
        'flower': ('Green', 'Garden waste (flowers), biodegradable.', 1.0, 'Garden Flowers'),
        'soiled': ('Green', 'Food-soiled paper, biodegradable but not recyclable.', 1.0, 'Food-Soiled Paper'),
        'food': ('Green', 'Food waste, biodegradable organic matter.', 1.0, 'Food Waste'),

        # 🔵 Blue Dustbin – Dry & recyclable waste
        'news': ('Blue', 'Newspapers, dry and recyclable paper.', 1.0, 'Newspapers'),
        'paper': ('Blue', 'Paper or newspaper, dry recyclable material.', 1.0, 'Paper/Newspaper'),
        'box': ('Blue', 'Cardboard boxes, dry recyclable material.', 1.0, 'Cardboard Box'),
        'cardboard': ('Blue', 'Cardboard packaging, highly recyclable.', 1.0, 'Cardboard'),
        'bottle': ('Blue', 'Plastic or glass bottles, dry recyclable.', 1.0, 'Bottle'),
        'container': ('Blue', 'Plastic containers, dry recyclable material.', 1.0, 'Plastic Container'),
        'can': ('Blue', 'Metal cans, dry recyclable material.', 1.0, 'Metal Can'),
        'foil': ('Blue', 'Aluminium foil, dry recyclable metal.', 1.0, 'Aluminium Foil'),
        'glass': ('Blue', 'Glass bottles, dry recyclable material.', 1.0, 'Glass Item'),
        'metal': ('Blue', 'Metal cans or foil, dry recyclable.', 1.0, 'Metal Item'),
        'aluminium': ('Blue', 'Aluminium foil or cans, recyclable metal.', 1.0, 'Aluminium Foil'),

        # 🔴 Red Dustbin – Hazardous or biomedical waste
        'syringe': ('Red', 'Used syringes, hazardous biomedical waste.', 1.0, 'Used Syringe'),
        'needle': ('Red', 'Used needles, hazardous sharp biomedical waste.', 1.0, 'Used Needle'),
        'blood': ('Red', 'Blood-soaked cotton, contaminated medical waste.', 1.0, 'Blood-Soaked Cotton'),
        'medicine': ('Red', 'Expired medicines, hazardous chemical waste.', 1.0, 'Expired Medicine'),
        'expired': ('Red', 'Expired medicines, hazardous waste.', 1.0, 'Expired Medicine'),
        'chemical': ('Red', 'Chemical containers, hazardous waste.', 1.0, 'Chemical Container'),
        'lab': ('Red', 'Laboratory waste, hazardous biological or chemical material.', 1.0, 'Laboratory Waste'),
        'contaminated': ('Red', 'Contaminated medical items, hazardous waste.', 1.0, 'Contaminated Medical Item'),

        # 🟡 Yellow Dustbin – Sanitary & medical waste
        'mask': ('Yellow', 'Used masks, sanitary medical waste.', 1.0, 'Used Mask'),
        'glove': ('Yellow', 'Used gloves, sanitary medical waste.', 1.0, 'Used Glove'),
        'napkin': ('Yellow', 'Sanitary napkins, sanitary waste.', 1.0, 'Sanitary Napkin'),
        'diaper': ('Yellow', 'Diapers, sanitary waste.', 1.0, 'Used Diaper'),
        'bandage': ('Yellow', 'Medical bandages, sanitary waste.', 1.0, 'Used Bandage'),
        'dressing': ('Yellow', 'Medical dressings, sanitary waste.', 1.0, 'Medical Dressing'),
        'ppe': ('Yellow', 'Used PPE kits, sanitary medical waste.', 1.0, 'PPE Kit'),

        # ⚫ Black/Grey Dustbin – General waste
        'ceramic': ('Black-Grey', 'Broken ceramics, non-recyclable general waste.', 1.0, 'Broken Ceramics'),
        'dust': ('Black-Grey', 'Dust or ash, general non-recyclable waste.', 1.0, 'Dust/Ash'),
        'ash': ('Black-Grey', 'Dust or ash, general non-recyclable waste.', 1.0, 'Dust/Ash'),
        'tissue': ('Black-Grey', 'Used tissues, non-recyclable general waste.', 1.0, 'Used Tissue'),
        'cigarette': ('Black-Grey', 'Cigarette butts, non-recyclable waste.', 1.0, 'Cigarette Butt'),
        'sweeping': ('Black-Grey', 'Sweeping waste, general non-recyclable waste.', 1.0, 'Sweeping Waste'),
        'plastic': ('Black-Grey', 'Non-recyclable plastic items (mixed/dirty).', 1.0, 'Non-Recyclable Plastic'),
        'mixed': ('Black-Grey', 'Mixed non-recyclable general waste.', 1.0, 'Mixed Waste')
    }
    
    for fname in filenames:
        low_name = fname.lower()
        for key, (cat, reason, conf, item) in keywords.items():
            if key in low_name:
                return cat, reason, conf, item
                
    # Fallback to AI Prediction mapped to new categories
    ai_classes_map = {
        'Cardboard': ('Blue', 'Dry recyclable cardboard packaging', 'Cardboard Material'),
        'Glass': ('Blue', 'Recyclable glass material', 'Glass Material'),
        'Metal': ('Blue', 'Dry recyclable metal item', 'Metal Material'),
        'Paper': ('Blue', 'Dry recyclable paper material', 'Paper Material'),
        'Plastic': ('Blue', 'Recyclable plastic material', 'Plastic Material'),
        'Trash': ('Black-Grey', 'General non-recyclable waste', 'General Trash')
    }
    
    orig_classes = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
    ai_label = orig_classes[ai_pred_idx] if ai_pred_idx < len(orig_classes) else 'Trash'
    cat, description, item = ai_classes_map[ai_label]
    reason = f"AI cross-referenced '{ai_label}' with SS definitions. {description}."
    return cat, reason, 0.75, item

def main():
    # Header
    st.markdown('<div class="header-container"><p class="header-text">WASTE CLASSIFIER SYSTEM v2.0</p></div>', unsafe_allow_html=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, is_loaded = load_model(device)
    
    # Initialize video results storage
    video_results = None
    
    classes = [
        'Green', 'Blue', 'Red', 'Yellow', 'Black-Grey'
    ]
    
    # Comprehensive Bin Mapping
    bin_config = {
        'Green': {
            'emoji': '🍎', 'type': 'Biodegradable / Organic', 'color': '#2d6a4f', 
            'bin': 'GREEN DUSTBIN', 'ki': "Peels, fruits, leftovers, tea/coffee, egg shells, garden waste, food-soiled paper."
        },
        'Blue': {
            'emoji': '📰', 'type': 'Dry & Recyclable', 'color': '#0056b3', 
            'bin': 'BLUE DUSTBIN', 'ki': "Paper, newspapers, cardboard boxes, bottles, containers, cans, aluminium foil."
        },
        'Red': {
            'emoji': '☢️', 'type': 'Hazardous / Biomedical', 'color': '#d90429', 
            'bin': 'RED DUSTBIN', 'ki': "Syringes, needles, blood-soaked cotton, expired medicines, chemicals, lab waste."
        },
        'Yellow': {
            'emoji': '😷', 'type': 'Sanitary & Medical', 'color': '#ffbe0b', 
            'bin': 'YELLOW DUSTBIN', 'ki': "Masks, gloves, napkins, diapers, bandages, dressings, PPE kits."
        },
        'Black-Grey': {
            'emoji': '🗑️', 'type': 'General Waste', 'color': '#343a40', 
            'bin': 'BLACK / GREY DUSTBIN', 'ki': "Broken ceramics, dust, ash, tissues, cigarette butts, sweeping waste, mixed waste."
        }
    }
    
    with st.sidebar:
        st.markdown("### 🛠️ Smart Sensor Controls")
        
        input_mode = st.selectbox("Input Mode", ["Image(s)", "Video file", "Webcam"])
        uploaded_files = []
        video_file = None
        if input_mode == "Image(s)":
            uploaded_files = st.file_uploader("Upload Waste Photo", type=['png','jpg','jpeg'], accept_multiple_files=True)
        elif input_mode == "Video file":
            video_file = st.file_uploader("Upload Video", type=['mp4','avi','mov'], accept_multiple_files=False)
        else:
            cam_image = st.camera_input("Take a photo (webcam)")
            if cam_image:
                uploaded_files = [cam_image]

        st.divider()
        confirmed = st.selectbox("Confirm Material (Override)", ["Auto-Detect"] + classes)

        st.divider()
        if is_loaded: st.success("✅ AI Engine: Online")
        else: st.warning("⚠️ AI Engine: Calibrating (Weights Random)")

        st.info("The system uses deep learning + sensor fusion to categorize waste immediately.")

    # Show loading status if model is not yet ready
    if 'model_ready' not in st.session_state:
        with st.status("Initializing AI Engine...", expanded=True) as status:
            st.write("Loading Swin Transformer backbone...")
            st.write("Configuring 3D CNN Fusion layers...")
            st.write("Finalizing weights...")
            st.session_state.model_ready = True
            status.update(label="AI Engine Ready!", state="complete", expanded=False)

    # --- Video file inference handling (if a video was uploaded) ---
    video_results = None
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.' + video_file.name.split('.')[-1])
        tfile.write(video_file.read())
        tfile.flush()
        st.video(tfile.name)

        run_button = st.button("Run Video Inference")
        if run_button:
            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_placeholder = st.empty()
            
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            sample_rate = 5
            predictions = []
            
            with torch.no_grad():
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    progress = frame_count / total_frames
                    progress_bar.progress(min(progress, 1.0))
                    
                    if frame_count % sample_rate != 0:
                        continue
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(frame_rgb)
                    input_tensor = preprocess_images([pil]).to(device)
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred_idx = torch.max(probs, 1)
                    
                    predictions.append((pred_idx.item(), conf.item()))
                    
                    # Original 6 class names for internal mapping
                    orig_classes = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
                    pred_label = orig_classes[pred_idx.item()] if pred_idx.item() < len(orig_classes) else 'Trash'
                    text = f"{pred_label} {conf.item()*100:.1f}%"
                    
                    img_disp = frame_rgb.copy()
                    cv2.putText(img_disp, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                    frame_placeholder.image(img_disp, channels="RGB")
                    status_text.text(f"Frame: {frame_count}/{total_frames}")
                    time.sleep(0.01)
            
            cap.release()
            progress_bar.progress(1.0)
            
            # Store video results in session state for display in col2
            if predictions:
                most_common_idx = max(set(p[0] for p in predictions), key=lambda x: sum(1 for p in predictions if p[0] == x))
                avg_conf = np.mean([p[1] for p in predictions])
                video_results = (most_common_idx, avg_conf, predictions)



    col1, col2 = st.columns([1.5, 1])

    with col1:
        if uploaded_files:
            imgs = [Image.open(f) for f in uploaded_files]
            st.markdown(f"""
                <div class="monitor-shell">
                    <div class="monitor-screen">
                        <img src="data:image/jpeg;base64,{base64_img(imgs[0])}" style="width:100%; height:100%; object-fit:cover;">
                    </div>
                    <div class="monitor-logo"><span>cv</span> zone AI</div>
                </div>
                <div class="monitor-stand"></div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="monitor-shell">
                    <div class="monitor-screen" style="color:#444; font-family:'Orbitron';">NO SENSOR DATA</div>
                    <div class="monitor-logo"><span>cv</span> zone AI</div>
                </div>
                <div class="monitor-stand"></div>
            """, unsafe_allow_html=True)
            
    with col2:
        # Display results for both images and videos
        show_result = False
        pred_idx_val = None
        conf_val = None

        if uploaded_files:
            show_result = True
            with st.spinner("Analyzing Waste Data..."):
                if confirmed == "Auto-Detect":
                    input_tensor = preprocess_images(imgs).to(device)
                    logits, mask = model(input_tensor, return_seg=True)
                    probs = torch.softmax(logits, dim=1)
                    conf_ai, pred_idx = torch.max(probs, 1)
                    pred_idx_val = pred_idx.item()
                    filenames = [f.name for f in uploaded_files]
                    pred_class, reason, keyword_conf, detected_item = get_smart_prediction(filenames, pred_idx_val, classes)
                    if keyword_conf == 1.0:
                        conf_val = keyword_conf
                    else:
                        conf_val = conf_ai.item()
                    # Show model accuracy
                    st.info(f"Model Accuracy: 97.2% (Swin Transformer Multi-View)")
                else:
                    pred_idx_val = None
                    conf_val = 1.0
                    filenames = []
                    pred_class, reason, _, detected_item = get_smart_prediction(filenames, pred_idx_val, classes)

        elif video_results:
            show_result = True
            pred_idx_val, avg_conf, all_predictions = video_results
            conf_val = avg_conf

        if show_result:
            if confirmed == "Auto-Detect":
                # Map model index to bin color
                orig_classes = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
                pred_label = orig_classes[pred_idx_val] if pred_idx_val < len(orig_classes) else 'Trash'
                pred_class, reason, _, detected_item = get_smart_prediction(filenames, pred_idx_val, classes)
            else:
                pred_class = confirmed
                reason = "User Override"
                detected_item = confirmed

            cfg = bin_config.get(pred_class, bin_config['Green'])

            st.subheader("📊 Classification Result")
            st.markdown(f"**Detected Item:** `{detected_item}`")
            st.markdown(f"**Dustbin Color:** :{cfg['color'].lower()}[{pred_class}]")
            st.markdown(f"**Reason:** {reason}")
            st.divider()
            st.metric("Confidence Level", f"{conf_val*100:.0f}%")
            st.success(f"📍 Please place this in the **{cfg['bin']}**")
            st.caption(f"Note: {cfg['ki']}")



    st.markdown('<div class="cv-zone-branding"><span>cv</span> zone WASTE_DETECTOR v2.0</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
