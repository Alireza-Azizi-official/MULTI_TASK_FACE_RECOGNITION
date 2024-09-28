import numpy as np 
import tensorflow as tf 
from keras._tf_keras.keras.models import Sequential, Model
from keras._tf_keras.keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D
from keras._tf_keras.keras.optimizers import Adam, SGD
import os
import cv2
import csv
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2 
from keras._tf_keras.keras.applications import ResNet50, VGG19, VGG16, InceptionResNetV2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import face_recognition 

# print(os.getcwd())

path = ('/content/drive/MyDrive')

#X_train_path
train_path = os.path.join(path, "NIR_face_detection","train")
train_images = os.listdir(train_path)

#X_val_path
valid_path = os.path.join(path, "NIR_face_detection","valid")
valid_images = os.listdir(valid_path)

#X_test_path
test_path = os.path.join(path, "NIR_face_detection","test")
test_images = os.listdir(test_path)

#y_train_path (labels)
train_labels_path = os.path.join(path, "NIR_face_detection", "trainlabels.csv")
train_labels_df = pd.read_csv(train_labels_path)

#y_val_path (labels)
valid_labels_path = os.path.join(path, "NIR_face_detection", "validlabels.csv")
valid_labels_df = pd.read_csv(valid_labels_path)

#y_test_path (labels)
test_labels_path = os.path.join(path, "NIR_face_detection", "testlabels.csv")
test_labels_df = pd.read_csv(test_labels_path)

# visuelise data
def match_bounding_boxes(img_path, labels_df):
    image_filename = img_path.split('/')[-1]

    # Filter the labels DataFrame to find matching rows
    matching_rows = labels_df[labels_df['filename'] == image_filename]

    # Extract bounding box coordinates
    bounding_boxes = []
    for _, row in matching_rows.iterrows():
        x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        bounding_boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1))) # Convert coordinates to integers

    return bounding_boxes

for i in range (0, 4):
  image_path = os.path.join (train_path, train_images[i])
  detected_faces = match_bounding_boxes(image_path, train_labels_df)
  print(detected_faces)
  image = cv2.imread(image_path)

  # Draw bounding boxes on the image
  for x, y, w, h in detected_faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
  cv2.imshow(image)

cv2.waitKey(0)
cv2.destroyAllWindows()

def scale_bounding_box(bbox, original_dims, new_dims):
  x_min, y_min, x_max, y_max = bbox
  original_width, original_height = original_dims
  new_width, new_height = new_dims

  x_scale = new_width / original_width
  y_scale = new_height / original_height

  new_x_min = int(x_min * x_scale)
  new_y_min = int(y_min * y_scale)
  new_x_max = int(x_max * x_scale)
  new_y_max = int(y_max * y_scale)

  return (new_x_min, new_y_min, new_x_max, new_y_max)

def process_images_and_bboxes(image_dir, annotation_file, new_dims=(224, 224)):
  scaled_bboxes = {}
  with open(annotation_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row

    for row in reader:
      image_name = row[0]  # Assuming image name is in the first column

      if len(row) >= 5 and all(cell.isdigit() for cell in row[1:5]):
        bbox = tuple(map(int, row[1:5]))  # Assuming bbox coordinates are in columns 2-5
        image_path = os.path.join(image_dir, image_name)

        if os.path.exists(image_path):
          with Image.open(image_path) as img:
            original_dims = img.size
          scaled_bbox = scale_bounding_box(bbox, original_dims, new_dims)
          scaled_bboxes[image_name] = scaled_bbox
        else:
          print(f"Image not found: {image_path}")
      else:
        print(f"Skipping row: {row}")

  return scaled_bboxes

scaled_bboxes = process_images_and_bboxes(train_path, train_labels_path)
test_scaled_bbox = process_images_and_bboxes(test_path, test_labels_path)
val_scaled_bbox = process_images_and_bboxes(valid_path, valid_labels_path)

# #checking example
# for image_name, bbox in scaled_bboxes.items():
#   image_path = os.path.join(train_path, image_name)
#   if os.path.exists(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (224, 224))  # Resize the image
#     x_min, y_min, x_max, y_max = bbox
#     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Draw bounding box

#     # Convert BGR (OpenCV) to RGB (Matplotlib)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     plt.imshow(img)
#     plt.title(image_name)
#     plt.show()
#   else:
#     print(f"Image not found: {image_path}")


#train_data_preproccessing
X_train = []  # Image data
y_train = []  # Bounding box labels

for image_name, bbox in scaled_bboxes.items():
  image_path = os.path.join(train_path, image_name)
  if os.path.exists(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize the image

    # Normalize image data
    img = img / 255.0

    X_train.append(img)
    y_train.append(bbox)
  else:
    print(f"Image not found: {image_path}")

#valid_data_preproccessing
X_val = []
y_val = []

for image_name, bbox in val_scaled_bbox.items():
  image_path = os.path.join(valid_path, image_name)
  if os.path.exists(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize the image

    # Normalize image data
    img = img / 255.0
    X_val.append(img)
    y_val.append(bbox)
  else:
    print(f"Image not found: {valid_path}")

#test_data_preproccessing
X_test = []
y_test = []

for image_name, bbox in test_scaled_bbox.items():
  image_path = os.path.join(test_path, image_name)
  if os.path.exists(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize the image

    # Normalize image data
    img = img / 255.0

    X_test.append(img)
    y_test.append(bbox)
  else:
    print(f"Image not found: {test_path}")


# Convert lists to NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array (X_val)
y_val = np.array (y_val)
X_test = np.array (X_test)
y_test = np.array (y_test)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(X_val.shape)
print(y_val.shape)

#Classification_data_preproccessing
y_train_class = []
y_val_class = []
y_test_class = []

#test_data_preproccessing
for image in train_images:
  if image.endswith(".jpg"):
    image_path = os.path.join(train_path, image)
    labels_df = pd.read_csv(train_labels_path)
    matching_rows = labels_df[labels_df['filename'] == image]
    classes = []
    for _, row in matching_rows.iterrows():
        classes.append(row['class'])
    y_train_class.append(classes)

  else:
    print(f"Skipping non-image file: {image}")

#valid_data_preproccessing
for image in valid_images:
  if image.endswith(".jpg"):
    image_path = os.path.join(valid_path, image)
    labels_df = pd.read_csv(valid_labels_path)
    mathcing_rows = labels_df[labels_df['filename'] == image]
    classes = []
    for _, row in mathcing_rows.iterrows():
        classes.append(row['class'])
    y_val_class.append(classes)


  else:
    print(f"Skipping non-image file: {image}")

#test_data_preproccessing
for image in test_images:
  if image.endswith(".jpg"):
    image_path = os.path.join(test_path, image)
    labels_df = pd.read_csv(test_labels_path)
    mathcing_rows = labels_df[labels_df['filename'] == image]
    classes = []
    for _, row in mathcing_rows.iterrows():
        classes.append(row['class'])
    y_test_class.append(classes)


  else:
    print(f"Skipping non-image file: {image}")

# Convert lists to NumPy arrays
y_train_class = np.array(y_train_class, dtype=object)
y_val_class = np.array (y_val_class, dtype=object)
y_test_class = np.array (y_test_class, dtype=object)

print(f'y_train_class shape: {y_train_class.shape}')
print(f'y_val_class shape: {y_val_class.shape}')
print(f'y_test_class shape: {y_test_class.shape}')

#OneHotEncoding the classification data
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Flatten the lists of lists
y_train_class = np.concatenate(y_train_class)
y_val_class = np.concatenate(y_val_class)
y_test_class = np.concatenate(y_test_class)

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Reshape y_train_class to a 2D array with one column
y_train_class = encoder.fit_transform(y_train_class.reshape(-1,1))
y_val_class = encoder.transform(y_val_class.reshape(-1,1))
y_test_class = encoder.transform(y_test_class.reshape(-1,1))

print(y_train_class.shape)
print(y_val_class.shape)
print(y_test_class.shape)

base_model = VGG16 (weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers[-5:]:
  layer.trainable = True

model = base_model.output
model = GlobalAveragePooling2D()(model)
model = Dense(1024, activation='relu')(model)

# Number of classes (face and non-face)
class_output = Dense(2, activation='sigmoid', name='class_output')(model)

# Bounding box predictions
bbox_output = Dense(4, activation='linear', name='bbox_output')(model)

model = Model(inputs=base_model.input, outputs=[class_output, bbox_output])

# Compile the model
model.compile(optimizer= Adam(learning_rate=0.0001),
              loss={'class_output': 'binary_crossentropy',
                    'bbox_output': 'mse'},
               metrics={'class_output': 'accuracy',
                       'bbox_output': 'mae'})
model.summary()

history = model.fit(X_train,
         {'bbox_output': y_train, 'class_output': y_train_class},
         batch_size=32,
         epochs=100,
          validation_data=(X_val, {'class_output': y_val_class, 'bbox_output': y_val})
        )

predictions = model.predict(X_test, batch_size=32)

acc = history.history['class_output_accuracy']
val_acc = history.history['val_class_output_accuracy']
print(acc)
print(val_acc)
print(predictions)
bbox_predictions = predictions [1]
face_predictions = predictions [0]

from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

# Convert face_predictions to binary predictions based on a threshold (e.g., 0.5)
face_predictions_binary = (face_predictions > 0.5).astype(int)

accuracy = accuracy_score(y_test_class, face_predictions_binary)
print(f'Accuracy: {accuracy}')

# Calculate MAE
mae = mean_absolute_error(y_test, bbox_predictions)
print(f'Mean Absolute Error: {mae}')

# Calculate MSE
mse = mean_squared_error(y_test, bbox_predictions)
print(f'Mean Squared Error: {mse}')

# Calculate RMSE
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def draw_bounding_box(image, bbox, color='r', linewidth=2):
  """Draws a bounding box on an image.

  Args:
    image: The image to draw on (NumPy array).
    bbox: Bounding box coordinates [x_min, y_min, x_max, y_max].
    color: Color of the bounding box (default: red).
    linewidth: Width of the bounding box lines (default: 2).
  """

  # Create a Rectangle patch
  rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                           linewidth=linewidth, edgecolor=color, facecolor='none')

  # Create figure and axes
  fig, ax = plt.subplots(1)

  # Display the image
  ax.imshow(image)

  # Add the patch to the Axes
  ax.add_patch(rect)

  plt.show()

image = X_test[0]
predicted_bbox = bbox_predictions[0]
y_test_bbox = y_test[0]

# Draw the bounding box on the image
draw_bounding_box(image, y_test_bbox)
draw_bounding_box(image, predicted_bbox)

print(y_test_bbox)
print(predicted_bbox)
print(mean_absolute_error (y_test_bbox, predicted_bbox))

model = model.save('modelfinal.h5')
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
model = tf.keras.models.load_model('/content/drive/MyDrive/modelfinal.h5', custom_objects={'mse': mean_squared_error})

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def draw_bounding_box(image, bbox, color='r', linewidth=2):
  """Draws a bounding box on an image.

  Args:
    image: The image to draw on (NumPy array).
    bbox: Bounding box coordinates [x_min, y_min, x_max, y_max].
    color: Color of the bounding box (default: red).
    linewidth: Width of the bounding box lines (default: 2).
  """

  # Create a Rectangle patch
  rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                           linewidth=linewidth, edgecolor=color, facecolor='none')

  # Create figure and axes
  fig, ax = plt.subplots(1)

  # Display the image
  ax.imshow(image)

  # Add the patch to the Axes
  ax.add_patch(rect)

  plt.show()
  
  image = cv2.imread ('/content/drive/MyDrive/5.jpg')
image = cv2.resize(image, (224, 224))
image = image / 255.0
image = np.array(image)

imagepresiction = model.predict(np.expand_dims(image, axis=0))
print (imagepresiction[1])

predicted_bbox = imagepresiction[1][0] # Extract the bounding box from the prediction
acc_prediction = imagepresiction[0]
print (acc_prediction)
draw_bounding_box(image, predicted_bbox)

image = cv2.imread ('/content/drive/MyDrive/6.jpg')
image = cv2.resize(image, (224, 224))
image = image / 255.0
image = np.array(image)

imagepresiction = model.predict(np.expand_dims(image, axis=0))
print (imagepresiction[1])

predicted_bbox = imagepresiction[1][0] # Extract the bounding box from the prediction
acc_prediction = imagepresiction[0]
print (acc_prediction)
draw_bounding_box(image, predicted_bbox)

if acc_prediction[0][0] > 0.5:
  print ("Face Detected")
else:
  print ("Face Not Detected")


def draw_bounding_box(image, bbox, color='r', linewidth=2):
  """Draws a bounding box on an image and crops the detected face.

  Args:
    image: The image to draw on (NumPy array).
    bbox: Bounding box coordinates [x_min, y_min, x_max, y_max].
    color: Color of the bounding box (default: red).
    linewidth: Width of the bounding box lines (default: 2).
  """

  # Create a Rectangle patch
  rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                           linewidth=linewidth, edgecolor=color, facecolor='none')

  # Create figure and axes
  fig, ax = plt.subplots(1)

  # Display the image
  ax.imshow(image)

  # Add the patch to the Axes
  ax.add_patch(rect)

  plt.show()

  # Crop the face using the bounding box coordinates
  x_min, y_min, x_max, y_max = map(int, bbox) # Convert coordinates to integers
  cropped_face = image[y_min:y_max, x_min:x_max]

  # Display the cropped face
  plt.imshow(cropped_face)
  plt.show()
  cv2.imwrite('cropped_face.jpg', cropped_face * 255.0) # Multiply by 255.0 to restore original pixel values
  
  
image = cv2.imread ('/content/drive/MyDrive/6.jpg')
image = cv2.resize(image, (224, 224))
image = image / 255.0
image = np.array(image)

imagepresiction = model.predict(np.expand_dims(image, axis=0))
print (imagepresiction[1])

predicted_bbox = imagepresiction[1][0] # Extract the bounding box from the prediction
acc_prediction = imagepresiction[0]
print (acc_prediction)
draw_bounding_box(image, predicted_bbox)

# Access the desired element of the array for comparison
if acc_prediction[0][0] > 0.5:
  print ("Face Detected")
else:
  print ("Face Not Detected")
  


# Load NIR image
nir_face_image =  '/content/cropped_face.jpg'
nir_face_image = cv2.imread(nir_face_image)

#Load visible image
visible_face_image =  '/content/drive/MyDrive/vis2.png'
visible_face_image = cv2.imread(visible_face_image)

nir_face_encodings = face_recognition.face_encodings(nir_face_image)
# Check if any faces were detected
if nir_face_encodings:
  nir_face_encoding = nir_face_encodings[0]
  visible_face_encodings = face_recognition.face_encodings(visible_face_image)
  if visible_face_encodings:
    visible_face_encoding = visible_face_encodings[0]

    # Compare encodings
    results = face_recognition.compare_faces([nir_face_encoding], visible_face_encoding)
    distance = face_recognition.face_distance([nir_face_encoding], visible_face_encoding)

    if results[0]:
        print("Faces match.")
    else:
        print("Faces do not match.")

    print("Distance:", distance)
  else:
    print("No faces found in visible image")
else:
  print("No faces found in NIR image")