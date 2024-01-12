# main.py

from A.SVM_model import TaskA_SVM
from A.CNN_model import TaskA_CNN
from B.CNN_model_B import TaskB_CNN

def main():
        
    # Specify the path to your dataset
    pneumonia_path = 'Datasets/pneumoniamnist.npz'
    pathminst_path ='Datasets/pathmnist.npz'

    # Create an instance for each
    task_a_SVM= TaskA_SVM(pneumonia_path)
    task_a_CNN = TaskA_CNN(pneumonia_path)
    task_b_CNN = TaskB_CNN(pathminst_path)

    while True:

        # Display menu    
        print("MACHINE LEARNING ASSIGNMENT:")
        print("WELCOME")

        print("Please Select an option:")
        print("0. Visualise Datasets")
        print("1. Run Task A - Train SVM without PCA")
        print("2. Run Task A - Train SVM with PCA")
        print("3. Run Task A - Train CNN")
        print("4. Run Task A - Use Pre-trained CNN (Recommended)")
        print("5. Run Task B - Train CNN (Recommended)")
        print("6. Run Task B - Use Pre-trained CNN (Recommended)")
        print("7. Exit")

        # Get user input
        choice = input("Enter your choice (1-7): ")

        # Perform tasks based on user's choice
        if choice == '0':
            # Call the method 
            task_a_SVM.visualise_dataset()
            task_b_CNN.visualise_dataset()

        elif choice == '1':
            # TASK A - runs SVM no PCA
            task_a_SVM.train_SVM_model()

        elif choice == '2':
            # TASK A - runs SVM with PCA
            task_a_SVM.train_SVM_model_pca()

        elif choice == '3':
            # TASK A - CNN not trained
            task_a_CNN.train_CNN_model()

        elif choice == '4':
            # TASK A - CNN pre trained
            task_a_CNN.test_pre_trained_model() #pretrained

        elif choice == '5':
            # TASK B - CNN not trained
            task_b_CNN.train_CNN_model()

        elif choice == '6':
            # TASK B - CNN pre trained
            task_b_CNN.test_pre_trained_model() #pretrained

        elif choice == '7':
            print("Thank you for grading my assignment! Goodbye!")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 6.")


if __name__ == "__main__":
    main()