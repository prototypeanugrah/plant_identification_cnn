from plant_classifier.entities.load_data import load_data


def data_pipeline():
    """
    Data pipeline.
    """
    train_data, validation_data, test_data, id2label = load_data()

    return train_data, validation_data, test_data, id2label
