import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor, InterpolationMode
from torchvision.models import vit_h_14, ViT_H_14_Weights
from PIL import Image
from flask import Flask, request, jsonify
import io
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import timm
import torch.nn as nn
from functools import partial
from collections import OrderedDict

# --- Initialize Flask app ---
app = Flask(__name__)

# --- Model filenames/config ---
CATARACT_MODEL_FILENAME = 'cataract_model.pth'
DIABETES_MODEL_DIR = 'diabetes_classifier_savedmodel' 
BMI_MODEL_FILENAME = 'bmi_model.pt'
IMG_SIZE = (224, 224)

# Diabetes model class names (matching training)
DIABETES_CLASS_NAMES = [
    'diabetic_acanthosis_nigricans',
    'diabetic_skin_tags',
    'diabetic_xanthalesma',
    'non_diabetic'
]

# --- Load Cataract Model (PyTorch) ---
cataract_model = torchvision.models.resnet50(weights=None)
num_ftrs = cataract_model.fc.in_features
cataract_model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_ftrs, 1),
    torch.nn.Sigmoid()
)
try:
    cataract_model.load_state_dict(torch.load(CATARACT_MODEL_FILENAME, map_location=torch.device('cpu')))
    cataract_model.eval()
    print(f"PyTorch model '{CATARACT_MODEL_FILENAME}' loaded successfully!")
except Exception as e:
    print(f"Error loading PyTorch model: {e}")
    cataract_model = None

# --- Load Diabetes Model (TensorFlow) ---
try:
    diabetes_model_layer = TFSMLayer(DIABETES_MODEL_DIR, call_endpoint='serving_default')
    print("TensorFlow SavedModel loaded as TFSMLayer")
except Exception as e:
    print(f"Error loading TensorFlow SavedModel as TFSMLayer: {e}")
    diabetes_model_layer = None

# --- Load Anemia Model ---
try:
    face_expert_model = tf.keras.models.load_model("face_expert_model.keras")
    lips_expert_model = tf.keras.models.load_model("lips_expert_model.keras")
    eye_expert_model = tf.keras.models.load_model("eye_expert_model.keras")
    print("Anemia expert models loaded successfully!")
except Exception as e:
    print(f"Error loading anemia models: {e}")
    face_expert_model = None
    lips_expert_model = None
    eye_expert_model = None

# --- Load BMI Model ---
class BMIHead(nn.Module):
    def __init__(self):
        super(BMIHead, self).__init__()
        self.linear1 = nn.Linear(1280, 640)
        self.linear2 = nn.Linear(640, 320)
        self.linear3 = nn.Linear(320, 160)
        self.linear4 = nn.Linear(160, 80)
        self.linear5 = nn.Linear(80, 1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.gelu(x)
        x = self.linear3(x)
        x = self.gelu(x)
        x = self.linear4(x)
        x = self.gelu(x)
        x = self.linear5(x)
        out = self.gelu(x)
        return out

def get_bmi_model():
    # We load the weights from the pre-trained ViT-H-14 model
    model = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)
    
    # We freeze the model's parameters as the original code did
    for param in model.parameters():
        param.requires_grad = False
        
    # We replace the classification head with our custom BMIHead
    model.heads = BMIHead()
    
    return model

bmi_model = None
try:
    bmi_model = get_bmi_model()
    state_dict = torch.load(BMI_MODEL_FILENAME, map_location=torch.device('cpu'))
    
    # This is often needed when a model is saved within a larger class or a dictionary.
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('model.heads.') else k 
        if 'linear' in name or 'gelu' in name or 'dropout' in name:
            name = f'heads.{name}' 
        new_state_dict[name] = v

    bmi_model.load_state_dict(state_dict)
    bmi_model.eval()

    print(f"PyTorch BMI model '{BMI_MODEL_FILENAME}' loaded successfully!")
except Exception as e:
    print(f"Error loading PyTorch BMI model: {e}")
    bmi_model = None

# --- Image preprocessing for cataract model ---
preprocess_cataract = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                         std=[0.2023, 0.1994, 0.2010])
])


# --- Image preprocessing for diabetes model ---
def preprocess_diabetes_image(image):
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    return img_array.astype(np.float32)

# --- Image preprocessing for anemia model ---
import numpy as np
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def get_bbox(face_landmarks, points, img_w, img_h):
    x_coords = [int(face_landmarks.landmark[p].x * img_w) for p in points]
    y_coords = [int(face_landmarks.landmark[p].y * img_h) for p in points]
    x = min(x_coords)
    y = min(y_coords)
    w = max(x_coords) - x
    h = max(y_coords) - y
    return x, y, w, h

def preprocess_crop_for_expert(crop):
    if crop.size == 0:
        return None
    IMG_SIZE = 128
    resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    preprocessed = preprocess_input(resized)
    return np.expand_dims(preprocessed, axis=0)

# --- Image preprocessing for BMI model ---
preprocess_bmi = transforms.Compose([
    transforms.Resize([518], interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop([518]),
    transforms.ToTensor(), # Add ToTensor as it's not in the original vit_transforms
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --- Prediction endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if cataract_model is None or diabetes_model_layer is None or \
       face_expert_model is None or lips_expert_model is None or eye_expert_model is None:
        return jsonify({'error': 'One or more models not loaded!'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # --- Cataract prediction code here (unchanged) ---
        input_tensor = preprocess_cataract(image)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = cataract_model(input_batch)
            cataract_prob = torch.sigmoid(output[0]).item()
        cataract_pred = "Cataract Detected" if cataract_prob > 0.5 else "No Cataract Detected"
        cataract_confidence = cataract_prob * 100

        # --- Diabetes prediction code here (unchanged) ---
        img_array = preprocess_diabetes_image(image)
        diabetes_output = diabetes_model_layer(img_array)
        output_tensor = list(diabetes_output.values())[0]
        diabetes_preds = output_tensor.numpy()[0]
        diabetic_idx = np.argmax(diabetes_preds)
        diabetic_class = DIABETES_CLASS_NAMES[diabetic_idx] if diabetic_idx < len(DIABETES_CLASS_NAMES) else "Unknown"
        diabetic_confidence = diabetes_preds[diabetic_idx] * 100
        diabetes_pred = "Non-Diabetic" if diabetic_class == 'non_diabetic' else \
            f"Diabetic: {diabetic_class.replace('_',' ').title()} detected"

        # --- Anemia expert models processing ---

        # Convert to OpenCV BGR numpy for mediapipe and ROI extraction
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = face_mesh.process(image_np)

        if not results.multi_face_landmarks:
            anemia_result = "Face not detected for anemia analysis"
        else:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = image_np.shape

            # Landmarks for lips, eyes, face regions (same points as your frontend)
            lip_points = [61, 291, 13, 14]
            eye_points = [33, 246, 161, 144, 466, 263, 388, 373]
            face_points = [10, 234, 454, 152]

            padding = 10

            def safe_crop(x, y, w_, h_):
                x1 = max(x - padding, 0)
                y1 = max(y - padding, 0)
                x2 = min(x + w_ + padding, image_np.shape[1])
                y2 = min(y + h_ + padding, image_np.shape[0])
                return image_np[y1:y2, x1:x2]

            lip_bbox = get_bbox(face_landmarks, lip_points, w, h)
            eye_bbox = get_bbox(face_landmarks, eye_points, w, h)
            face_bbox = get_bbox(face_landmarks, face_points, w, h)

            lip_crop = safe_crop(*lip_bbox)
            eye_crop = safe_crop(*eye_bbox)
            face_crop = safe_crop(*face_bbox)

            input_lips = preprocess_crop_for_expert(lip_crop)
            input_eye = preprocess_crop_for_expert(eye_crop)
            input_face = preprocess_crop_for_expert(face_crop)

            lip_score = lips_expert_model.predict(input_lips)[0][0] if input_lips is not None else 0
            eye_score = eye_expert_model.predict(input_eye)[0][0] if input_eye is not None else 0
            face_score = face_expert_model.predict(input_face)[0][0] if input_face is not None else 0

            combined_score = 0.25 * eye_score + 0.25 * lip_score + 0.5 * face_score
            if combined_score > 0.5:
                anemia_result = f"Anemia signs detected. Confidence: {combined_score*100:.2f}%"
            else:
                anemia_result = f"No anemia signs detected. Confidence: {combined_score*100:.2f}%"

        # --- BMI prediction code here ---
        input_tensor_bmi = preprocess_bmi(image)
        input_batch_bmi = input_tensor_bmi.unsqueeze(0)
        with torch.no_grad():
            bmi_pred_tensor = bmi_model(input_batch_bmi)
            bmi_value = float(bmi_pred_tensor.item())

        if bmi_value < 18.5:
            bmi_risk = "Low (Underweight)"
        elif bmi_value < 25:
            bmi_risk = "Low (Healthy)"
        elif bmi_value < 30:
            bmi_risk = "Medium (Overweight)"
        else:
            bmi_risk = "High (Obese)"
        
        bmi_result = f"Predicted BMI: {bmi_value:.2f} (Risk: {bmi_risk})"

        # --- Return JSON with all predictions ---
        return jsonify({
            'cataract_prediction': f"{cataract_pred} (Confidence: {cataract_confidence:.2f}%)",
            'diabetes_prediction': f"{diabetes_pred} (Confidence: {diabetic_confidence:.2f}%)",
            'anemia_prediction': anemia_result,
            'bmi_prediction': bmi_result
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Failed to process image'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
