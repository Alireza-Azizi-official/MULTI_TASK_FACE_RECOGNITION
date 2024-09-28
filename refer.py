import torch
import cv2
import os
from main import SiameseNetwork, preprocess

model = SiameseNetwork().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model.eval()  

reference_images_folder = r'C:\Users\Administrator\Desktop\face_recognition2_finished\application_data\verification_images'

reference_images = [os.path.join(reference_images_folder, f) for f in os.listdir(reference_images_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

embeddings = []

for image_path in reference_images:
    reference_image = preprocess(cv2.imread(image_path)).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  
    
    with torch.no_grad():
        reference_output = model.forward_one(reference_image)
    
    embeddings.append(reference_output)

torch.save(embeddings, 'reference_template.pth')
print("Reference templates saved successfully!")
