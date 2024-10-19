# CNN 
### Image classification in Pytorch using TinyVGG architecture

This application implements imgae classification on 6 randomly selected images from the input dataset.

**Example output:**

"""

True label names:

['Pullover', 'Pullover', 'T-shirt/top', 'Trouser', 'Bag', 'Sneaker']


Predicted label names:

['Pullover', 'Coat', 'T-shirt/top', 'Trouser', 'Bag', 'Sneaker']

"""


To run using docker:
- `docker build -t cnnimg .` 
- `docker run -it --rm cnnimg`

Alternatively, to train and test the model from the scratch:
- `python train.py`
- `python inference.py --test_path data/fashion_mnist_test.pt`
