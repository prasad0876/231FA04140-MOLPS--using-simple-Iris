from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_model():
    # Load Iris dataset
    iris = load_iris()

    X = iris.data
    y = iris.target

    # Split dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy, X_test, y_test, predictions


if __name__ == "__main__":
    model, accuracy, X_test, y_test, predictions = train_model()

    print("====================================")
    print("Machine Learning Model")
    print("====================================")
    print("Model   : Random Forest Classifier")
    print("Dataset : Iris Dataset")
    print(f"Accuracy: {accuracy:.2%}")
    print("Sample Predictions:", predictions[:5])
    print("====================================")
