#!/usr/bin/env python
# coding: utf-8

# # Brain Tumor Classification using CNN
# - **Label**:
#     - no_tumor
#     - meningioma_tumor
#     - pituitary_tumor
#     - glioma_tumor
# - **Dataset Link** :https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
# 

# ## Import Libraries 

# In[1]:


# Imports the necessary libraries for file handling and image processing
import os
import numpy as np




# Imports for loading and converting images to arrays (for Keras)
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Imports the function to split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# For one-hot encoding of labels (used in multi-class classification)
from tensorflow.keras.utils import to_categorical

# CNN Model architecture (for building Convolutional Neural Networks)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

# For plotting accuracy and loss graphs to visualize model performance
import matplotlib.pyplot as plt

# Early stopping callback to prevent overfitting during training
from keras.callbacks import EarlyStopping

# Import the Python Imaging Library (PIL) for advanced image handling
from PIL import Image


# ## Defining Class Labels for Tumor Categories

# In[3]:


# Define the folders and assign numerical labels to each tumor category
folders = {
    'no_tumor': 0,            # Label 0 for images with no tumor
    'meningioma_tumor': 1,    # Label 1 for meningioma tumor images
    'pituitary_tumor': 2,     # Label 2 for pituitary tumor images
    'glioma_tumor': 3         # Label 3 for glioma tumor images
}


# ## Initializing Image Size and Data Lists

# In[5]:


# Set the desired target size for each image (width, height) for uniform input into the model
image_size = (150, 150)

# Initialize empty lists to hold image data (X) and corresponding labels (Y)
X = []  # Will store the image arrays (features)
Y = []  # Will store the labels corresponding to each image


# In[8]:


#Printing Folder Names and Corresponding Labels
for folder, label in folders.items():
    print(f"Folder: {folder}, Label: {label}")


# ##  Loading Images and Labels from Folders with Image Count

# In[10]:


# Iterate through each folder and its assigned label
for folder, label in folders.items():
    # Create the full folder path for each tumor type
    folder_path = os.path.join(r"D:\Esraa Taher 2\Teaching\Deep Learning(SC 2024)\brain-tumor-dataset\Training", folder)
   
    # Count the number of images in the folder
    total_images = len(os.listdir(folder_path))
    print(f"Folder '{folder}' contains {total_images} images.")  # Print the total number of images in the folder
    
    # Loop through all image files in the current folder
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)  # Create the full image file path
        
        # Load the image and resize it to the target size (150x150)
        img = load_img(img_path, target_size=image_size)
        
        # Convert the image to an array and normalize pixel values by dividing by 255.0
        img_array = img_to_array(img) / 255.0
        
        # Append the image array to the X list (features)
        X.append(img_array)
        
        # Append the label (e.g., 0, 1, 2, 3) to the Y list (labels)
        Y.append(label)


# In[11]:


print(X[5])


# ## Displaying Sample Images from the Dataset

# In[12]:


import matplotlib.pyplot as plt

# Assuming X is a NumPy array containing images
# Let's print the first 5 images in X

for i in range(5):  # Adjust the range to display more or fewer images
    plt.imshow(X[i])  # Display the ith image
    plt.title(f'Image {i+1} - Label: {Y[i]}')  # Add a title to the plot
    plt.axis('off')  # Hide the axis for better visualization
    plt.show()  # Display the image


# In[13]:


# Convert to numpy arrays
X = np.array(X)  # Convert the list of images (X) to a NumPy array for efficient processing
Y = np.array(Y)  # Convert the list of labels (Y) to a NumPy array

# One-hot encode the labels
Y = to_categorical(Y, num_classes=len(folders))  # Convert integer labels to one-hot encoded format

# Split into training and validation sets (90% training, 10% validation)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)  


# ## Building a Convolutional Neural Network (CNN) Archiecture

# In[15]:


# Initialize the Sequential model
model = Sequential()

# Define the input layer explicitly
model.add(Input(shape=(150, 150, 3)))  # Input shape for images: 150x150 pixels, 3 color channels (RGB)

# First convolutional layer (no input_shape needed since Input layer is defined)
model.add(Conv2D(32, (3, 3), activation='relu'))  # 32 filters, 3x3 kernel, ReLU activation function
model.add(MaxPooling2D(pool_size=(2, 2)))  # Downsample the feature maps by 2
model.add(Dropout(0.25))  # Reduce overfitting by dropping 25% of the neurons

# Second convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))  # 64 filters, 3x3 kernel, ReLU activation
model.add(MaxPooling2D(pool_size=(2, 2)))  # Downsample the feature maps by 2
model.add(Dropout(0.25))  # Further reduce overfitting by dropping 25% of the neurons

# Third convolutional layer
model.add(Conv2D(128, (3, 3), activation='relu'))  # 128 filters, 3x3 kernel, ReLU activation
model.add(MaxPooling2D(pool_size=(2, 2)))  # Downsample the feature maps by 2

# Flatten the feature maps to feed into a dense layer
model.add(Flatten())  # Convert 2D feature maps into 1D feature vectors

# Dense layer with Dropout
model.add(Dense(128, activation='relu'))  # Fully connected layer with 128 neurons
model.add(Dropout(0.5))  # Dropout layer to reduce overfitting (50%)

# Output layer for classification
model.add(Dense(len(folders), activation='softmax'))  # Number of classes corresponds to the length of 'folders'

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  
# Using Adam optimizer and categorical crossentropy loss for multi-class classification

# Summary of the model architecture
model.summary()  # Print the model summary to view the architecture and number of parameters


# # Show summery table
# - first parameter in output shape refer to batch size
# - second parameter in output shape for height
# - third parameter in output shape for width
# - forth parameter in output shape for no of feature maps
# - param # refer to no of trainable parameters (weights and bias)
# $$
# Output Size = (\frac{{Input \, Size - Kernel \, Size + 2 \times Padding}}{{Stride}} + 1)
# $$
# - ex)conv2d (Conv2D) (None, 126, 126, 32) 896
# - 126 x 126 $$
# Output Size = (\frac{{128 - 3 + 2 \times 0}}{{1}} + 1)
# $$
# - 32 --> feature map after applying 32 filter
# - param# --> size of filter (width,height) *no of filter * no of channels +bias for each filter
# - 896 -->3 x 3 x 32 x 3 +32=896
# 
# 
# 

# ## Training the CNN Model with Early Stopping

# In[23]:


# Define Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Monitor the validation loss, stop training if it doesn't improve for 3 epochs, and restore the best weights.

# Train the model
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),  # Use validation data for evaluation
    epochs=10,  # Number of epochs to train
    callbacks=[early_stopping]  # Use early stopping and model checkpoint callbacks
)


# ##  Plotting Training and Validation Metrics

# In[25]:


# Plot Training and Validation Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')  # Plot training accuracy
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # Plot validation accuracy
plt.title('Accuracy per Epoch')  # Title for the accuracy plot
plt.xlabel('Epoch')  # Label for the x-axis
plt.ylabel('Accuracy')  # Label for the y-axis
plt.legend()  # Display legend to differentiate lines
plt.show()  # Show the accuracy plot

# Plot Training and Validation Loss
plt.plot(history.history['loss'], label='Training Loss')  # Plot training loss
plt.plot(history.history['val_loss'], label='Validation Loss')  # Plot validation loss
plt.title('Loss per Epoch')  # Title for the loss plot
plt.xlabel('Epoch')  # Label for the x-axis
plt.ylabel('Loss')  # Label for the y-axis
plt.legend()  # Display legend to differentiate lines
plt.show()  # Show the loss plot


# ## Loading and Preparing Test Images for Evaluation

# In[27]:


# Initialize empty lists to store test images and their corresponding labels
X_test = []  # List to hold the test images
Y_test = []  # List to hold the labels for the test images

# Load the test images and labels from the respective folders
for folder, label in folders.items():  # Iterate through each folder and its label
    # Construct the full path to the test images folder for the current category
    folder_path = os.path.join(r"D:\Esraa Taher 2\Teaching\Deep Learning(SC 2024)\brain-tumor-dataset\Testing", folder)
    
    # Check if the folder exists to avoid errors
    if os.path.exists(folder_path):
        # Iterate through each image in the current folder
        for img_name in os.listdir(folder_path):  # List all images in the folder
            img_path = os.path.join(folder_path, img_name)  # Construct the full image path
            img = load_img(img_path, target_size=image_size)  # Load the image and resize it
            img_array = img_to_array(img) / 255.0  # Convert the image to an array and normalize the pixel values to [0, 1]
            X_test.append(img_array)  # Append the normalized image to the X_test list
            Y_test.append(label)  # Append the corresponding label to the Y_test list
    else:
        # Print a message if the folder is not found
        print(f'Folder not found: {folder_path}')

# Convert the lists to NumPy arrays for further processing
X_test = np.array(X_test)  # Convert the list of images to a NumPy array
Y_test = np.array(Y_test)  # Convert the list of labels to a NumPy array


# ## Making Predictions and Evaluating Model Performance on Test Images

# In[29]:


# Make predictions on the test images using the trained model
predictions = model.predict(X_test)  # Get the predicted probabilities for each class

# Convert the predicted probabilities to class labels
predicted_classes = np.argmax(predictions, axis=1)  # Get the index of the highest probability for each prediction

# Since Y_test already contains class indices, there's no need to use np.argmax here
true_classes = Y_test  # Assign true_classes to the actual labels from Y_test

# Compare the predicted classes with the true labels
correct_predictions = np.sum(predicted_classes == true_classes)  # Count how many predictions are correct
total_predictions = len(true_classes)  # Get the total number of predictions
incorrect_predictions = total_predictions - correct_predictions  # Calculate the number of incorrect predictions

# Print out the results
print(f'Total images: {total_predictions}')  # Display the total number of test images
print(f'Correctly classified images: {correct_predictions}')  # Display the count of correct classifications
print(f'Incorrectly classified images: {incorrect_predictions}')  # Display the count of incorrect classifications


# In[30]:


# Calculate the accuracy of the model on the test dataset
accuracy = correct_predictions / total_predictions  # Compute accuracy as the ratio of correct predictions to total predictions

# Print the accuracy as a percentage, formatted to two decimal places
print(f'Model accuracy on the test set: {accuracy * 100:.2f}%')  # Display the accuracy percentage


# ## Visualizing the Confusion Matrix for Model Predictions

# In[32]:


# Import necessary libraries for numerical operations, plotting, and metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report

# Calculate the confusion matrix to evaluate the model's performance
cm = confusion_matrix(true_classes, predicted_classes)  # Compare true and predicted classes

# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(10, 7))  # Set the figure size
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',  # Create a heatmap with annotations
            xticklabels=['No tumor 0', 'Meningioma tumor 1', 'Pituitary tumor 2', 'Glioma tumor 3'],  # X-axis labels
            yticklabels=['No tumor 0', 'Meningioma tumor 1', 'Pituitary tumor 2', 'Glioma tumor 3'])  # Y-axis labels
plt.xlabel('Predicted Label')  # Label for the X-axis
plt.ylabel('True Label')  # Label for the Y-axis
plt.title('Confusion Matrix')  # Title of the plot
plt.show()  # Display the plot


# In[33]:


# Print classification report
report = classification_report(true_classes, predicted_classes, target_names=['No tumor', 'Meningioma tumor', 'Pituitary tumor', 'Glioma tumor'])
print(report)


# In[34]:


from tensorflow.keras.callbacks import ModelCheckpoint

# Define the ModelCheckpoint callback with the `.keras` extension
mc = ModelCheckpoint(r"D:\Esraa Taher 2\Teaching\Deep Learning(SC 2024)\best_model.keras", monitor='val_loss', save_best_only=True)


# In[35]:


from keras.models import load_model
saved_model = load_model(r"D:\Esraa Taher 2\Teaching\Deep Learning(SC 2024)\best_model.keras")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




