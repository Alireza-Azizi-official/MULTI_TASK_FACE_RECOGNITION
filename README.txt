This application is a comprehensive multitask face recognition system that integrates four different recognition capabilities:

Normal Face Recognition
Thermal Face Recognition
Deep Fake Recognition
Infrared Recognition
The project leverages multiple technologies such as os, sys, cv2, random, torch, numpy, PIL (Pillow), Keras, TensorFlow, and Kivy to achieve its functionalities.

Project Overview
This application employs a deep learning approach to detect and verify faces using a variety of datasets and models. It includes both traditional Convolutional Neural Networks (CNN) and Siamese Networks to provide robust recognition across different modes.

Dataset Structure
The system requires a substantial dataset of more than 10 million images. However, due to GitHub's storage limitations, the dataset is not included in this repository. You can create your dataset using the following structure:

positive: Images of the target person to be recognized.
anchor: Different images of the same target person.
negative: Images of other people for contrastive learning.
When capturing a photo using OpenCV, the system compares the image with the dataset. If the distance between the captured image and the anchor images is below a specific threshold, the system verifies the person; otherwise, it's marked as unverified.

Distance Measurement and Model Choice
The app uses L1Dist to calculate the distance between feature vectors. The Siamese Network model is employed for training, which has proven more effective than VGG16 or VGG19 in this context. A CNN is used for training with the following parameters:

Epochs: 50
Learning Rate: 0.00001
Patience: 5 (adjustable)
Model Training and Optimization
The model is saved after training to be reused in the final Kivy app, reducing processing time since it doesn't need to retrain for each image capture.

Data Augmentation
Data augmentation techniques are employed to enhance the variety of the dataset, ensuring the model generalizes better.

Early Stopping
An early stopping mechanism is used to halt training if the model's performance doesn't improve over 5 consecutive epochs, helping to prevent overfitting and save training time.

Labeling
For simplicity and efficiency, the dataset is labeled using binary labels (0 for unverified, 1 for verified).

Hardware Recommendation
Given the scale of the dataset (10 million+ images), it's strongly recommended to use a GPU for training:

GPU: RTX 3060
CPU: AMD Ryzen 9
RAM: 16GB
Using this setup, training takes approximately 9 hours.

Application Workflow
The final Kivy app loads all four trained models to handle the different recognition tasks:

Each time you click a button, it calls the corresponding model to perform the verification process.
The app includes a frame that changes color based on the verification status:
Green: Verified
Red: Unverified
The status label also changes color and text according to the verification result.
Technical Overview
Two recognition tasks are trained using TensorFlow models.
The other two tasks utilize the Siamese Network model.
This architecture allows seamless integration and efficient recognition across the different recognition modes.

Conclusion
This multitask face recognition application is built to handle complex recognition tasks efficiently using state-of-the-art deep learning techniques. Feel free to adjust model parameters or dataset structure as needed to enhance performance.