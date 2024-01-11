import numpy as np
from random import randint
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, auc


class TaskA_SVM:
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

    def visualise_dataset(self):
        # Concatenate training and validation datasets for better visualization
        all_images = np.concatenate([self.training_images, self.validation_images])
        all_labels = np.concatenate([self.training_labels, self.validation_labels])
        all_labels_flat = all_labels.flatten()
        
        # Check the number of labels per class
        class_counts = np.bincount(all_labels_flat)
        class_0_indices = np.where(all_labels_flat == 0)[0]
        class_1_indices = np.where(all_labels_flat == 1)[0]
        plt.figure(figsize=(8, 3))

        # Subplot for an image from Class 0
        plt.subplot(1, 3, 1)
        random_index_0 = randint(0, len(class_0_indices) - 1)
        plt.imshow(all_images[class_0_indices[random_index_0]], cmap='gray')
        plt.title('Class 0 (No Pneumonia)')

        # Subplot for an image from Class 1
        plt.subplot(1, 3, 2)
        random_index_1 = randint(0, len(class_1_indices) - 1)
        plt.imshow(all_images[class_1_indices[random_index_1]], cmap='gray')
        plt.title('Class 1 (Pneumonia)')

        # Display a bar graph showing the distribution of images per class
        # Subplot for the bar graph
        ax = plt.subplot(1, 3, 3)
        classes = ['0', '1']
        bars = ax.bar(classes, class_counts, color=['lightblue', 'lightblue'], edgecolor='black', linewidth=1)
        # Add numbers on top of each bar
        for bar, count in zip(bars, class_counts):
            plt.text(bar.get_x() + bar.get_width() / 2 - 0.05, bar.get_height() + 20, str(count), ha='center', va='bottom', fontsize=10)
        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Images')
        ax.set_title('Class Distribution')
        # Remove y-axis ticks
        ax.tick_params(axis='y', which='both', left=False, labelleft = False)
        # Remove the outside grid
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
    
        plt.tight_layout()
        plt.show()

    def preprocessing (self): 
        # Normalize pixel values to the range [0, 1]
        normalized_training_images = self.all_training_images.astype('float32') / 255.0

        #reshape images into 2D array
        normalized_training_images = normalized_training_images.reshape(len(normalized_training_images), -1)
        #reshape labels to a 1D array
        all_training_labels_flat = np.ravel(self.all_training_labels)

        return normalized_training_images, all_training_labels_flat

    def train_SVM_model(self):
        processed_training_images, processed_labels = self.preprocessing()

        # Define the parameter grid to search
        # param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['rbf', 'sigmoid'], 'gamma': [0.001, 0.01, 0.1]}
        param_grid = {'C': [10], 'kernel': ['rbf'], 'gamma': [0.1]}
        
        # Create an SVM model
        svm_model = SVC(tol=1e-3, max_iter=-1) #learning criterium
        
        # Create GridSearchCV object
        grid_search = GridSearchCV(svm_model, param_grid, scoring='accuracy',cv = 5, verbose = 1)
        print('grid_search')

        grid_search.fit(processed_training_images, processed_labels)

        # Print the mean accuracy for each combination of hyperparameters
        mean_test_scores = grid_search.cv_results_['mean_test_score']
        params = grid_search.cv_results_['params']
        for score, param in zip(mean_test_scores, params):
            print(f'Parameters: {param}, Mean Accuracy: {score}')

        # Print the best parameters found by the grid search
        print("Best Parameters:", grid_search.best_params_)
        # Get the best SVM model
        self.best_svm_model = grid_search.best_estimator_

        #now call the method to test the model on the testing set
        self.test_model()

    def test_model(self):
        
        #preprocessing the testing data images
        testing_images_normalized= self.testing_images.astype('float32') / 255.0
        test_predictions = self.best_svm_model.predict(testing_images_normalized.reshape(len(testing_images_normalized), -1))

        # Evaluate accuracy
        test_accuracy = accuracy_score(self.testing_labels, test_predictions)
        print(f'Test Accuracy: {test_accuracy}')

        # Evaluate Precision, Recall, F-1 and 
        print('Classification Report:')
        print(classification_report(self.testing_labels, test_predictions, zero_division = 1))

        # Calculate AUC Score
        fpr, tpr, thresholds = roc_curve(self.testing_labels, test_predictions)
        roc_auc = auc(fpr, tpr)
        print(f'AUC on testing set: {roc_auc:.4f}')

        # Plot ROC curve
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.4f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

        #print confusion matrix
        print('Confusion Matrix:')
        print(confusion_matrix(self.testing_labels, test_predictions))

        # Plot Confusion Matrix
        # plot_confusion_matrix(self.best_svm_model, testing_images_normalized.reshape(len(testing_images_normalized), -1), self.testing_labels, display_labels=['Class 0', 'Class 1'], cmap=plt.cm.Blues, normalize=None)
        # plt.title('Confusion Matrix')
        # plt.show()

        # # # Plot ROC Curve wrong??
        # plot_roc_curve(self.best_svm_model, testing_images_normalized.reshape(len(testing_images_normalized), -1), self.testing_labels)
        # plt.title('ROC Curve')
        # plt.show()

    def train_SVM_model_pca(self):
        # Normalize pixel values to the range [0, 1]
        # normalized_training_images = self.all_training_images.astype('float32') / 255.0
        # testing_images_normalized= self.testing_images.astype('float32') / 255.0
        #reshape images 
        normalized_training_images = self.all_training_images.reshape(len(self.all_training_images), -1)
        normalized_testing_images = self.testing_images.reshape(len(self.testing_images), -1)

        #Apply StandardScaler
        self.scaler = StandardScaler()
        normalized_training_images_scaled = self.scaler.fit_transform(normalized_training_images)
        normalized_testing_images_scaled = self.scaler.transform(normalized_testing_images)

        # Apply PCA
        self.pca = PCA(n_components=0.95)
        normalized_training_images_pca = self.pca.fit_transform(normalized_training_images_scaled)
        normalized_testing_images_pca = self.pca.transform(normalized_testing_images_scaled)

        #reshape labels to a 1D array
        all_training_labels_flat = np.ravel(self.all_training_labels)
        
        # Define the parameter grid to search
        # param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['rbf', 'sigmoid'], 'gamma': [0.001, 0.01, 0.1]}
        param_grid = {'C': [10], 'kernel': ['rbf'], 'gamma': [0.001]}


        # Create an SVM model
        # svm_model = SVC() #learning criterium
        svm_model = SVC(tol=1e-3, max_iter=-1)
        
        # Create GridSearchCV object
        grid_search = GridSearchCV(svm_model, param_grid, scoring='accuracy',cv = 5, verbose = 1)
        print('grid_search')

        grid_search.fit(normalized_training_images_pca, all_training_labels_flat)

        # Print the mean accuracy for each combination of hyperparameters
        mean_test_scores = grid_search.cv_results_['mean_test_score']
        params = grid_search.cv_results_['params']

        for score, param in zip(mean_test_scores, params):
            print(f'Parameters: {param}, Mean Accuracy: {score}')

        # Print the best parameters found by the grid search
        print("Best Parameters:", grid_search.best_params_)

        # Get the best SVM model
        self.best_svm_model = grid_search.best_estimator_

        test_predictions = self.best_svm_model.predict(normalized_testing_images_pca)

        # Evaluate accuracy
        test_accuracy = accuracy_score(self.testing_labels, test_predictions)
        print(f'Test Accuracy: {test_accuracy}')

        # Evaluate other metrics
        print('Classification Report:')
        print(classification_report(self.testing_labels, test_predictions))

        print('Confusion Matrix:')
        print(confusion_matrix(self.testing_labels, test_predictions))

        # Plot Confusion Matrix
        # plot_confusion_matrix(self.best_svm_model, normalized_testing_images_pca, self.testing_labels, display_labels=['Class 0', 'Class 1'], cmap=plt.cm.Blues, normalize=None)
        # plt.title('Confusion Matrix')
        # plt.show()

        # Calculate AUC Score
        fpr, tpr, thresholds = roc_curve(self.testing_labels, test_predictions)
        roc_auc = auc(fpr, tpr)
        print(f'AUC on testing set: {roc_auc:.4f}')

        # Plot ROC curve
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.4f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

        # # Plot ROC Curve
        # plot_roc_curve(self.best_svm_model, testing_images_normalized.reshape(len(testing_images_normalized), -1), self.testing_labels)
        # plt.title('ROC Curve')

        # # Calculate AUC Score
        # auc_score = roc_auc_score(self.testing_labels, test_predictions)
        # print(f'AUC Score: {auc_score}')

        # plt.show()
