# CNN 
## Image classification in Pytorch using TinyVGG architecture

This application implements imgae classification on 6 randomly selected images from the input dataset.

# Docker 
To create and run docker container:
- `docker build -t cnnimg .` 
- `docker run -it --rm cnnimg`

# Train & Inference
To train the model from the scratch and do inference:
- `python train.py`
- `python inference.py --test_path data/fashion_mnist_test.pt`

or, to do inference on the already trained and saved model:
- `python inference.py --test_path data/fashion_mnist_test.pt`

## Example output

```
True label names:

['Pullover', 'Pullover', 'T-shirt/top', 'Trouser', 'Bag', 'Sneaker']

Predicted label names:

['Pullover', 'Coat', 'T-shirt/top', 'Trouser', 'Bag', 'Sneaker']
```