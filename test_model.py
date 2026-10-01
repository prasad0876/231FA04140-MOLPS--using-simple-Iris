from model import train_model


def test_model_accuracy():
    _, accuracy, _, _, _ = train_model()

    # Model should have at least 90% accuracy
    assert accuracy >= 0.90


def test_model_predictions():
    _, _, X_test, y_test, predictions = train_model()

    # Number of predictions should match test data
    assert len(predictions) == len(y_test)

    # There should be predictions
    assert len(predictions) > 0


def test_prediction_classes():
    _, _, _, _, predictions = train_model()

    # Iris has three classes: 0, 1 and 2
    assert all(prediction in [0, 1, 2] for prediction in predictions)
