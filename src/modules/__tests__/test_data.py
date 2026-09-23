from src.modules.read_data import read_data

# Read in data once to reference for the following tests.
train_data, val_data, test_data = read_data()

def test_input_data_groups():
    """Expect input data to be split into three groups, training, val, testing."""
    assert len(train_data) == 2
    assert len(val_data) == 2
    assert len(test_data) == 2

def test_input_data_shape():
    """Expect each input dataset in/out to have a consistent shape."""
    assert train_data[0].shape[0] == train_data[1].shape[0]
    assert val_data[0].shape[0] == val_data[1].shape[0]
    assert test_data[0].shape[0] == test_data[1].shape[0]

def test_input_img():
    """Expect each data point to be able to contain a 28x28 img."""
    img_size = 28 * 28
    assert train_data[0].shape[1] == img_size
