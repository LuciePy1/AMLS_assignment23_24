import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import save_model, load_model
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import seaborn as sns

class TaskB_CNN:
    #this function initilises the class instance and calls the load function
    def __init__(self, path):
        self.load_split(path)

    #this function loads the dataset from the path and separate accroding to train/validation/test
    def load_split (self, path):


       #loading the dataset 
        pathdataset = np.load(path)

        # Create variables for the images
        self.training_images = pathdataset['train_images']
        self.validation_images = pathdataset['val_images']
        self.testing_images = pathdataset['test_images']

        # Create variables for the labels
        self.training_labels = pathdataset['train_labels']
        self.validation_labels = pathdataset['val_labels']
        self.testing_labels = pathdataset['test_labels']

    def visualise_dataset(self):
        all_images = np.concatenate([self.training_images, self.validation_images])
        all_labels = np.concatenate([self.training_labels, self.validation_labels])
        all_labels_flat= all_labels.flatten()

        # Check the number of labels per class
        class_counts = np.bincount(all_labels_flat)
        print(all_labels_flat)
        print(class_counts)

        # Get the number of classes
        num_classes = len(class_counts)

        # Plotting the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(all_labels_flat, bins=np.arange(num_classes + 1) - 0.5, rwidth=0.8, align='mid', color='pink', edgecolor='black')
        plt.xlabel('Class')
        plt.ylabel('Number of Images')
        plt.title('Distribution of Labels in the Dataset PathMNIST')
        plt.xticks(range(num_classes))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

        # Get uniq ue class labels
        unique_classes = np.unique(all_labels_flat)

        # Plot one random image from each class
        plt.figure(figsize=(15, 5))
        for i, class_label in enumerate(unique_classes):
            # Find indices for the current class
            class_indices = np.where(all_labels_flat == class_label)[0]
            
            if len(class_indices) > 0:
                # Randomly choose one index from the current class
                random_index = np.random.choice(class_indices)
                image = all_images[random_index]
                
                plt.subplot(2, 5, i + 1)  # Adjust subplot layout if needed
                plt.imshow(image, cmap='gray')  # Assuming the images are grayscale, adjust cmap accordingly
                plt.title(f'Class {class_label}')
                plt.axis('off')

        plt.show()

    def preprocessing (self):
        # Normalize pixel values to be between 0 and 1
        self.training_images = self.training_images / 255.0
        self.validation_images = self.validation_images / 255.0
        self.testing_images = self.testing_images / 255.0

        #no need to reshape bc we are already 3D      

    def build_CNN_model(self):
        
        self.model = Sequential()

        #convolutional layer 1
        self.model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 3), padding = 'same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D((2,2)))

        #could add another convolution layer here with deeper depth
        self.model.add(Conv2D(64, (3, 3), padding = 'same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D((2, 2)))

        #could add another convolution layer here with deeper depth
        self.model.add(Conv2D(64, (3, 3), padding = 'same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D((2, 2)))

        #Flatten and fully connected layers for classification 
        self.model.add(Flatten())
        self.model.add(Dense(64, activation= 'relu'))
        self.model.add(Dense(9, activation ='softmax'))

    def train_CNN_model(self):

        self.preprocessing()
        self.build_CNN_model()
        self.model.compile(optimizer = Adam(), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics = ['accuracy'])
        self.model.summary()

        # Define the early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = self.model.fit(self.training_images, self.training_labels, epochs = 20, batch_size = 128, validation_data=(self.validation_images, self.validation_labels), verbose=1,  callbacks=[early_stopping])

        #self.model.save('B/taskB_pretrained_CNN_3.h5')

        # Plot training history (Accuracy)
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        # Plot training history (Loss)
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')

        plt.legend()

        # Display the plots
        plt.tight_layout()
        plt.show()

        self.test_model()

    def test_model(self):
        
        #Predict test set
        label_predicted = self.model.predict(self.testing_images)
        predicted_labels = np.argmax(label_predicted, axis=1)

        #Accuracy and other metrics
        test_accuracy = accuracy_score(self.testing_labels, predicted_labels)
        print(f'Test Accuracy: {test_accuracy}')
        print(classification_report(self.testing_labels,predicted_labels))
        
        # Confusion Matrix
        print('Confusion Matrix:')
        print(confusion_matrix(self.testing_labels, predicted_labels))       
        cm = confusion_matrix(self.testing_labels, predicted_labels)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True Labels')
        plt.show()


    def test_pre_trained_model(self):

        self.preprocessing()
        self.model= load_model('B/taskB_pretrained_CNN_2.h5')
        self.model.summary()

        #Predict test set
        label_predicted = self.model.predict(self.testing_images)
        predicted_labels = np.argmax(label_predicted, axis=1)

        #Accuracy and other metrics
        test_accuracy = accuracy_score(self.testing_labels, predicted_labels)
        print(f'Test Accuracy: {test_accuracy}')
        print(classification_report(self.testing_labels,predicted_labels))
        
        # Confusion Matrix
        print('Confusion Matrix:')
        print(confusion_matrix(self.testing_labels, predicted_labels))       
        cm = confusion_matrix(self.testing_labels, predicted_labels)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True Labels')
        plt.show()

