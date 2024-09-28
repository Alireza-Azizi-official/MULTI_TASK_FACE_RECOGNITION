import os
import sys
import cv2
import random
import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from kivy.uix.boxlayout import BoxLayout
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window
from PIL import Image as PILImage
from kivy.config import Config
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.losses import MeanSquaredError

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Config.set('input', 'wm_pen', 'None')

class L1Dist(nn.Module):
    def __init__(self):
        super(L1Dist, self).__init__()

    def forward(self, x1, x2):
        return torch.abs(x1 - x2)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=10) 
        self.conv2 = nn.Conv2d(128, 256, kernel_size=7)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=4)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=4)  

        self.fc1 = nn.Linear(1024 * 2 * 2, 8192)  
        self.dropout = nn.Dropout(0.4)  

        self.batch_norm1 = nn.BatchNorm2d(128)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.batch_norm3 = nn.BatchNorm2d(512)
        self.batch_norm4 = nn.BatchNorm2d(1024)

        self.fc2 = nn.Linear(8192, 1)
        self.sigmoid = nn.Sigmoid()

    def forward_one(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return x
    
Window.size = (800, 600)
Window.clearcolor = (0, 0, 0, 1)

torch_model = SiameseNetwork().to(device)
keras_model = None  # متغیر برای مدل Keras

def load_torch_model(model, load_path="siamese_model.pth"):
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict)  
    model.eval()
    print(f"PyTorch model loaded from {load_path}")
    return model

torch_model = load_torch_model(torch_model, load_path="siamese_model.pth")

def load_keras_model(model_path):
    global keras_model
    keras_model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
    print(f"Keras model loaded from {model_path}")
    return keras_model

# Open webcam
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Error: Camera could not be opened")

# Face detection with dynamic frame color
def detect_and_draw_face(frame, color=(0, 0, 255)):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

    return frame

# Preprocess image based on the model type
def preprocess(frame, model_type='torch'):
    if model_type == 'keras':
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0 
        frame = np.expand_dims(frame, axis=0) 
    else:
        frame = cv2.resize(frame, (100, 100))
        frame = frame / 255.0
        frame = np.transpose(frame, (2, 0, 1)) 
        frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)
        frame = frame.to(device)

    return frame

def verify_image(frame, model_type='torch'):
    preprocessed_frame = preprocess(frame, model_type)

    if model_type == 'keras':
        if keras_model is None:
            load_keras_model(r'C:\Users\Administrator\Desktop\face_recognition2_finished\NIR_Face_Detection.h5')
        
        prediction = keras_model.predict(preprocessed_frame)
        if isinstance(prediction, list):
            prediction = np.array(prediction[0])  

        if prediction.shape == ():  
            prediction = np.array([prediction])

        if np.any(prediction > 0.5): 
            return "Verified"
        else:
            return "Unverified"

    elif model_type == 'torch':
        with torch.no_grad():
            new_output = torch_model.forward_one(preprocessed_frame)  

        reference_embeddings = torch.load('reference_template.pth', map_location=device)
        distances = []
        for reference_output in reference_embeddings:
            distance = torch.norm(new_output - reference_output, p=1).item()
            distances.append(distance)

        min_distance = min(distances)

        if min_distance < 35.76:  
            return "Verified"
        else:
            return "Unverified"

class CamApp(App):
    def build(self):
        self.current_frame_color = (0, 0, 255)  
        
        main_layout = BoxLayout(orientation="horizontal", spacing=10, padding=20)
        left_layout = BoxLayout(orientation="vertical", size_hint=(0.3, 1), spacing=20, padding=[25, 20, 0, 0])
        right_layout = BoxLayout(orientation="vertical", size_hint=(0.7, 1))

        btn_style = {"size_hint": (1, None),
                     "background_color": (1, 255 / 255, 74/ 255, 1),
                     "color": (1, 1, 1, 1),
                     'font_size': 28,
                     'font_name': r'C:\Users\Administrator\Desktop\face_recognition2_finished\Font\times new roman bold.ttf',
                     'height': 170}

        self.verify_button = Button(text="Normal Recognition", on_press=self.verify, **btn_style)
        self.thermal_button = Button(text="Thermal Recognition", on_press=self.thermal_check, **btn_style)
        self.deepfake_button = Button(text="Deepfake Recognition", on_press=self.deepfake_check, **btn_style)
        self.infrared_button = Button(text="Infrared Recognition", on_press=self.infrared_check, **btn_style)

        exit_button_style = {"size_hint": (1, None),
                             "background_color": (128, 179, 210, 1),
                             "color": (0, 0, 0, 1),
                             "font_size": 60,
                             'font_name': r'C:\Users\Administrator\Desktop\face_recognition2_finished\Font\times new roman bold.ttf',                             
                             "height": 100}
        self.exit_button = Button(text="Exit", on_press=self.exit_app, **exit_button_style)

        left_layout.add_widget(self.verify_button)
        left_layout.add_widget(self.thermal_button)
        left_layout.add_widget(self.deepfake_button)
        left_layout.add_widget(self.infrared_button)
        left_layout.add_widget(self.exit_button)

        self.web_cam = KivyImage(size_hint=(1, 0.9), allow_stretch=True)
        right_layout.add_widget(self.web_cam)
        right_layout.add_widget(BoxLayout(size_hint=(1, 0.1)))

        self.verification_box = BoxLayout(size_hint=(1, None), padding=10, height=100)
        with self.verification_box.canvas.before:
            self.verification_box_color = Color(1, 1, 1, 1)
            self.rect = Rectangle(size=self.verification_box.size, pos=self.verification_box.pos)

        self.verification_box.bind(size=self.update_rect, pos=self.update_rect)
        self.verification_label = Label(text="STATUS", halign="center", valign="middle", color=(0, 0, 0, 1),
                                        size_hint=(1, None), height=100, font_size=40, font_name=r'Font\times new roman bold.ttf')
        self.verification_label.bind(size=self.verification_label.setter('text_size'))
        self.verification_box.add_widget(self.verification_label)
        right_layout.add_widget(self.verification_box)

        main_layout.add_widget(left_layout)
        main_layout.add_widget(right_layout)

        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return main_layout

    def update_rect(self, *args):
        self.rect.pos = self.verification_box.pos
        self.rect.size = self.verification_box.size

    def update(self, *args):
        ret, frame = capture.read()
        if not ret:
            print("Failed to capture frame")
            return
        
        frame_with_faces = detect_and_draw_face(frame, color=self.current_frame_color)
        frame_for_display = cv2.resize(frame_with_faces, (800, 600))
        buf = cv2.flip(frame_for_display, 0).tobytes()
        img_texture = Texture.create(size=(frame_for_display.shape[1], frame_for_display.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def verify(self, instance):
        ret, frame = capture.read()
        if not ret:
            self.verification_label.text = "Failed to capture frame"
            return
        result = verify_image(frame, model_type='torch')
        self.verification_label.text = result
        
        if result == "Verified":
            self.change_verification_color(0, 1, 0, 1)
            self.current_frame_color = (0, 255, 0)  
        else:
            self.change_verification_color(1, 0, 0, 1)
            self.current_frame_color = (0, 0, 255)  

    def infrared_check(self, instance):
        load_keras_model(r'C:\Users\Administrator\Desktop\face_recognition2_finished\NIR_Face_Detection.h5')
        ret, frame = capture.read()
        if not ret:
            self.verification_label.text = "Failed to capture frame"
            return
        result = verify_image(frame, model_type='keras')
        self.verification_label.text = result
        
        if result == "Verified":
            self.change_verification_color(0, 1, 0, 1)
            self.current_frame_color = (0, 255, 0)  
        else:
            self.change_verification_color(1, 0, 0, 1)
            self.current_frame_color = (0, 0, 255)  

    def change_verification_color(self, r, g, b, a):
        self.verification_box.canvas.before.clear()
        with self.verification_box.canvas.before:
            self.verification_box_color = Color(r, g, b, a)
            self.rect = Rectangle(size=self.verification_box.size, pos=self.verification_box.pos)

    def deepfake_check(self, instance):
        global keras_model
        keras_model = load_keras_model(r'C:\Users\Administrator\Desktop\face_recognition2_finished\xception_deepfake_model2.h5')
        ret, frame = capture.read()
        
        if not ret:
            self.verification_label.text = 'Failed to capture frame.'
            return

        preprocessed_frame = preprocess(frame, model_type='keras')
        prediction = keras_model.predict(preprocessed_frame)
        percentage = prediction[0][0] * 100
                
        if percentage > 0.050:  
            result = 'Verified'
            color = (0, 255, 0)  
            self.change_verification_color(0, 1, 0, 1)
        else:
            result = 'Unverified'
            color = (0, 0, 255)  
            self.change_verification_color(1, 0, 0, 1)
            
        self.current_frame_color = color
        self.verification_label.text = f"{result} ({percentage:.2f}%)"
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{result} ({percentage:.2f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        frame_for_display = cv2.resize(frame, (800, 600))
        buf = cv2.flip(frame_for_display, 0).tobytes()
        img_texture = Texture.create(size=(frame_for_display.shape[1], frame_for_display.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

            
    def thermal_check(self, instance):
        ret, frame = capture.read()
        
        if not ret:
            self.verification_label.text = 'Failed to capture frame.'
            return
        
        result = self.thermal_check_logic(frame, model_type='torch')  
        self.verification_label.text = result
        
        if result == 'Verified':
            self.change_verification_color(0, 1, 0, 1)
            self.current_frame_color = (0, 255, 0)  
        else:
            self.change_verification_color(1, 0, 0, 1)
            self.current_frame_color = (0, 0, 255)  

    def thermal_check_logic(self, frame, model_type='torch'):
        preprocessed_frame = preprocess(frame, model_type)
        
        if model_type == 'torch':
            with torch.no_grad():
                new_output = torch_model.forward_one(preprocessed_frame)
            reference_embeddings = torch.load(r'C:\Users\Administrator\Desktop\face_recognition2_finished\thermal\reference_template_thermal.pth', map_location=device)
            distances = []
            for reference_output in reference_embeddings: 
                distance = torch.norm(new_output - reference_output, p=1).item()
                distances.append(distance)
            min_distance = min(distances)
            print(min(distances), max(distances))  
            
            if min_distance < 1060:
                return 'Verified'
            else:
                return 'Unverified'


    def exit_app(self, instance):
        capture.release()  
        cv2.destroyAllWindows()
        App.get_running_app().stop()


if __name__ == "__main__":
    CamApp().run()
