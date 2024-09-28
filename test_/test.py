# کد اولیه 
import cv2 
import os 
import random
import numpy as np
from matplotlib import pyplot as plt 

from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from keras.metrics import Precision, Recall
import tensorflow as tf 
import uuid




POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')



# make the directories 
# os.makedirs(POS_PATH)
# os.makedirs(NEG_PATH)
# os.makedirs(ANC_PATH)

# move lfw images to the following repository data/negative
# for directory in os.listdir('lfw'):
#     for file in os.listdir(os.path.join('lfw', directory)):
#         EX_PATH = os.path.join('lfw', directory, file)
#         NEW_PATH  = os.path.join(NEG_PATH, file)
#         os.replace(EX_PATH, NEW_PATH)
        

# access webcam 
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read(4)
    
#     frame = frame[120:120 + 250, 200 :200 + 250, :]
#     plt.imshow(frame[120:120 + 250, 200 :200 + 250, :])
    
#     # collect anchors 
#     if cv2.waitKey(1) & 0XFF == ord('a'):
#         # create a unique file path 
#         imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
#         # write out anchor image 
#         cv2.imwrite(imgname, frame)
        
#     # collect positives 
#     if cv2.waitKey(1) & 0XFF == ord('p'):
#         # create a unique file path q
#         imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
#         # write out anchor image 
#         cv2.imwrite(imgname, frame)
        
    
#     cv2.imshow('Image Collection', frame)
    
#     if cv2.waitKey(1) & 0XFF == ord('q'):
#         break
    
# cap.release()
# cv2.destroyAllWindows()

## ----------------- newly added 
# def data_aug(img):
#     data = []
#     for i in range(9):
#         img = tf.image.stateless_random_brightness(img, max_delta = 0.02, seed = (1, 2))
#         img = tf.image.stateless_random_contrast(img, lower = 0.6, upper = 1, seed = (1, 3))
#         # img = tf.image.stateless_random_crop(img, size = (20, 20, 3), seed = (1, 2))
#         img = tf.image.stateless_random_flip_left_right(img, seed = (np.random.randint(100), np.random.randint(100)))
#         img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality = 90, max_jpeg_quality = 100, seed = (np.random.randint(100), np.random.randint(100)))
#         img = tf.image.stateless_random_saturation(img, lower = 0.9, upper = 1, seed = (np.random.randint(100), np.random.randint(100)))
        
#         data.append(img)
#     return data 

# ## ------------------ newly added2 
# img_path = os.path.join(ANC_PATH, 'f3651869-6b7d-11ef-b980-a036bc653070.jpg')
# img = cv2.imread(img_path)
# augmented_images = data_aug(img)

# for image in augmented_images:
#     cv2.imwrite(os.path.join(ANC_PATH, '{}.jpeg'.format(uuid.uuid1())), image.numpy())
    
# for file_name in os.listdir(os.path.join(POS_PATH)):
#     img_path = os.path.join(POS_PATH, file_name)
#     img = cv2.imread(img_path)
#     augmented_images = data_aug(img)
    
#     for image in augmented_images:
#         cv2.imwrite(os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())
        

# to have access to all the dataset  
anchor   = tf.data.Dataset.list_files(ANC_PATH + '\*.jpg').take(3000)
positive = tf.data.Dataset.list_files(POS_PATH + '\*.jpg').take(3000)
negative = tf.data.Dataset.list_files(NEG_PATH + '\*.jpg').take(3000)

# dir_test = anchor.as_numpy_iterator()
# print(dir_test.next())
# OUT PUT = b'data\\anchor\\e2d5d9b0-6a89-11ef-9479-a036bc653070.jpg'


# preprocessin - scale and resizing
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    
    # Decode the image
    img = tf.io.decode_jpeg(byte_img)
    
    # If the image is grayscale, ensure the last dimension is 1
    if len(img.shape) == 2:
        # Add the channel dimension to the grayscale image
        img = tf.expand_dims(img, axis=-1)

    # Check if the image is grayscale (has 1 channel) and convert to RGB
    if img.shape[-1] == 1:  # If it's a grayscale image
        img = tf.image.grayscale_to_rgb(img)
    
    # Resize the image to the desired size (100x100)
    img = tf.image.resize(img, (100, 100))
    
    # Scale the image values to be between 0 and 1
    img = img / 255.0
    
    return img

# print(preprocess('data\\anchor\\e2d5d9b0-6a89-11ef-9479-a036bc653070.jpg'))

# create labelled dataset 
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

# # check the labeled dataset
# print(tf.zeros(len(anchor)))
# print(tf.ones(len(anchor)))

samples = data.as_numpy_iterator()
example = (samples.next())   

# build train and test part 
def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

res  = preprocess_twin(*example)
# plt.imshow(res[7])
# plt.show()

# build dataloader pipeline 
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size = 10000)

# training part 
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# testing partition  
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

# building embedding layer 
def make_embedding():
    inp =Input(shape = (100, 100, 3), name ='input_iamge')
    
    # first block 
    c1 = Conv2D(68 , (10, 10), activation ='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding ='same')(c1)
    
    # second block 
    c2 = Conv2D(128, (7, 7), activation ='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding ='same')(c2)
    
    # third block
    c3 = Conv2D(128, (4, 4), activation ='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding ='same')(c3)
    
    # final embedding part
    c4 = Conv2D(256, (4, 4), activation = 'relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation ='sigmoid')(f1)
        
    embedding =  Model(inputs = [inp], outputs = [d1], name ='embedding')
    embedding.summary()
    return Model(inputs = [inp], outputs = [d1], name = 'embedding')

    
embedding = make_embedding()

# siamese L1 Distance class
class L1Dist(Layer):
    # Initialization of the layer
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    # Call method for forward pass
    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)

# Function to build the Siamese model
def make_siamese_model():
    # Define inputs
    input_image = Input(name='input_img', shape=(100, 100, 3))
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    # Get the embeddings for both images
    inp_embedding = embedding(input_image)
    val_embedding = embedding(validation_image)

    # Define the Siamese distance layer (L1 distance)
    siamese_layer = L1Dist()  # Define the L1Dist layer here
    siamese_layer._name = 'distance'

    # Calculate the L1 distance between the embeddings
    distances = siamese_layer([inp_embedding, val_embedding])

    # Add a classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    # Create the Siamese Network model
    siamese_network = Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

    return siamese_network

# Build the siamese model
siamese_model = make_siamese_model()

# Compile the siamese model
siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[Precision(), Recall()])

# loss and optimizer 
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)

# checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt = opt, siamese_model = siamese_model)

# train step function for one batch of data 
test_batch = train_data.as_numpy_iterator()
batch1 = test_batch.next()
x = batch1[:2]
y = batch1[2]
# print(np.array(x).shape)
# print(y)

# loss and optimizer 
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)  

# train step function for one batch of data 
@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        x = batch[:2]
        y = batch[2]
        yhat = siamese_model(x, training=True)
        loss = binary_cross_loss(y, yhat)
        
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    
    return loss, yhat

# build training loop 
def train(data, EPOCHS):
    for epoch in range(1, EPOCHS+1):
        print(f'\n EPOCH {epoch}/{EPOCHS}')
        progbar = tf.keras.utils.Progbar(len(data))
        
        # creating a metric object
        r = Recall()
        p = Precision()
        
        for idx, batch in enumerate(data):
            loss, yhat = train_step(batch)
            
            # Ensure yhat is not None before updating metrics
            if yhat is not None:
                r.update_state(batch[2], yhat)
                p.update_state(batch[2], yhat)
            
            progbar.update(idx+1)
        
        print(f'Loss: {loss.numpy()}, Recall: {r.result().numpy()}, Precision: {p.result().numpy()}')

        # Use CheckpointManager to manage multiple checkpoints
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

        # save checkpoints at every 10th epoch
        if epoch % 10 == 0:
            save_path = checkpoint_manager.save()
            print(f"Saved checkpoint for epoch {epoch}: {save_path}")

# train the model 
EPOCHS = 50 
train(train_data, EPOCHS)

# evaluate the model
# get a batch of test data 
test_input, test_val, y_true = test_data.as_numpy_iterator().next()

# make predictions 
y_hat = siamese_model.predict([test_input, test_val])

# post processing the results 
a = [1 if prediction > 0.5 else 0 for prediction in y_hat]
print(a)
print(y_true)
        
# creating a metric object
m = Recall()
# calculating the recall_value by passing the true labels and predicted labels separately
m.update_state(y_true, y_hat)
# return recall result 
recall_result = m.result().numpy()

print(f"Recall: {recall_result}")

# save model 
siamese_model.save('siamesemodelv2.h5')

# reload model
model = tf.keras.models.load_model('siamesemodelz.h5', custom_objects={'L1Dist': L1Dist})

# Compile the model after loading
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[Precision(), Recall()])

# Now, make predictions with the reloaded and compiled model
a = model.predict([test_input, test_val])

# View predictions
print(a)

# real time test
# verification function
for image in os.listdir(os.path.join('application_data', 'verification_images')):
    validation_img = preprocess(os.path.join('application_data', 'verification_images', image))
    # print(validation_img)
    
def verify(model, detection_threshold, verification_threshold):
    # build results  array 
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image))
        
        result = model.predict (list(np.expand_dims([input_img, validation_img], axis = 1))) 
        results.append(result)
        
    # detection threshold : metric obove which a prediction is considered positive
    detection = np.sum(np.array(results) > detection_threshold)
    # verification threshold : propotion of positive predictions / total positive samples
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
    verified = verification > verification_threshold
    return results, verified 

    
# opencv real time verification 
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120: 120 + 250, 200: 200 + 250, :]
    cv2.imshow('Verification', frame)
    
    # verification trigger 
    if cv2.waitKey(10) & 0xFF == ord('v'):
        # save input image to  application_data/input_image folder 
        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
        
        # run verification 
        # اگر خواستی دقت بیشتری بگیری این عددا رو کم و زیاد کن یا عکسای وریفیکیشن رو زیاد کن
        results, verified = verify(model, 0.9, 0.7)
        print(verified)
        
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

    