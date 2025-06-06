def split_data(data, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)

    return train_data, test_data


def save_splits(train_data, test_data, train_path, test_path):
    # Save the training and testing datasets to specified paths
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)