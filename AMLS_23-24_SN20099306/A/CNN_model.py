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
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns

class TaskA_CNN:
    #this function initilises the class instance and calls the load function
    def __init__(self, path):
        self.load_split(path)

    #this function loads the dataset from the path and separate accroding to train/validation/test
    def load_split (self, path):

        #loading the dataset 
        pneumoniadataset = np.load(path)

        # Splitting into trainning, validation and testing sets
        self.training_images = pneumoniadataset['train_images']
        self.validation_images = pneumoniadataset['val_images']
        self.testing_images = pneumoniadataset['test_images']

        # Combine training and validation 
        self.all_training_images = np.concatenate([self.training_images, self.validation_images])

        # Same for the labels
        self.training_labels = pneumoniadataset['train_labels']
        self.validation_labels = pneumoniadataset['val_labels']
        self.all_training_labels = np.concatenate([self.training_labels, self.validation_labels])
        self.testing_labels = pneumoniadataset['test_labels']


    def preprocessing (self): 
        #normalising 
        self.training_images = self.training_images.astype('float32') / 255.0
        self.validation_images = self.validation_images.astype('float32') / 255.0
        self.testing_images = self.testing_images.astype('float32') / 255.0

        #reshape to have three dimensions 
        self.training_images = np.expand_dims(self.training_images, axis = 3)
        self.validation_images = np.expand_dims(self.validation_images, axis = 3)
        self.testing_images = np.expand_dims(self.testing_images, axis = 3)

    def build_CNN_model(self):
        
        self.model = Sequential()

        #convolutional layer 1
        self.model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D((2,2)))

        #Flatten and fully connected layers for classification 
        self.model.add(Flatten())
        self.model.add(Dense(64, activation= 'relu'))
        self.model.add(Dense(1, activation ='sigmoid'))


    def train_CNN_model(self):
        
        #initialising the model
        self.preprocessing()
        self.build_CNN_model()
        self.model.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'])
        self.model.summary()

        #Early Stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the model with Early Stopping
        history = self.model.fit(
            self.training_images, self.training_labels,
            epochs=50, batch_size=32,
            validation_data=(self.validation_images, self.validation_labels),
            callbacks=[early_stopping],  # Early Stopping callback
            verbose=1
        )

        #self.model.save('A/taskA_pretrained_CNN_NEW.h5')

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

        #Evaluate on validation set
        test_loss, test_accuracy = self.model.evaluate(self.validation_images, self.validation_labels)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        
        #Predict test set
        label_predicted = self.model.predict(self.testing_images)

        #Accuracy and other metrics
        test_accuracy = accuracy_score(self.testing_labels, label_predicted.round())
        print(f'Test Accuracy: {test_accuracy}')
        print(classification_report(self.testing_labels,label_predicted.round()))
        
        # Confusion Matrix
        print('Confusion Matrix:')
        print(confusion_matrix(self.testing_labels, label_predicted.round()))       
        cm = confusion_matrix(self.testing_labels, label_predicted.round())
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 14})
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        # Calculate AUC score
        auc_score = roc_auc_score(self.testing_labels, label_predicted)
        print(f'AUC Score: {auc_score:.4f}')

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(self.testing_labels, label_predicted)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()


    def test_pre_trained_model(self):

        self.preprocessing()
        self.model= load_model('A/taskA_pretrained_CNN.h5')
        self.model.summary()

        #Evaluate on validation set
        test_loss, test_accuracy = self.model.evaluate(self.validation_images, self.validation_labels)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        
        #Predict test set
        label_predicted = self.model.predict(self.testing_images)

        #Accuracy and other metrics
        test_accuracy = accuracy_score(self.testing_labels, label_predicted.round())
        print(f'Test Accuracy: {test_accuracy}')
        print(classification_report(self.testing_labels, label_predicted.round()))
        
        # Confusion Matrix
        print('Confusion Matrix:')
        print(confusion_matrix(self.testing_labels, label_predicted.round()))       
        cm = confusion_matrix(self.testing_labels, label_predicted.round())
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, annot_kws={"size": 14})
        plt.title('Confusion Matrix - CNN')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()

        # Calculate AUC score
        auc_score = roc_auc_score(self.testing_labels, label_predicted)
        print(f'AUC Score: {auc_score:.4f}')

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(self.testing_labels, label_predicted)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()



       