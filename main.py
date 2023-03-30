from TPC1.dataset import Dataset
from TPC3.decisionTree import DecisionTrees
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # Load the dataset using the Dataset class
    data = Dataset.read(file_path="./datasets/titanic.csv", label="Survived")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size=0.2, random_state=2023)

    #Set threshold to 5% of the total number of instances
    threshold = int(0.05 * len(data.X)) 

    # Create a decision tree model
    clf = DecisionTrees(threshold=threshold)

    # Train the decision tree model using the training set
    clf.fit(X_train, y_train, X_test, y_test, depth=4)

    # Evaluate the accuracy of the model using the testing set
    #y_pred = clf.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)
    #print(f"Accuracy: {accuracy}")



    # Calcule a precisão da árvore antes da poda
    accuracy_before_pruning = clf.predict(X_test)

    # Faça a poda da árvore
    clf.reduced_error_pruning()

    # Calcule a precisão da árvore após a poda
    accuracy_after_pruning = clf.predict(X_test)

    # Compare as precisões antes e após a poda
    print(f"A precisão antes da poda foi de {accuracy_before_pruning:.2f}%")
    print(f"A precisão após a poda foi de {accuracy_after_pruning:.2f}%")