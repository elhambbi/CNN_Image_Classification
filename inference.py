import torch
import random
import os
import argparse
from train import cnnmodel1

# INFERENCE
def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare the sample (add a batch dimension and pass to target device)
            sample = torch.unsqueeze(sample, dim=0).to(device)

            # Forward pass (model outputs raw logits)
            pred_logit = model(sample)              #the number of logits=the number of classes for each image
            #print(pred_logit)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)     #squeeze to remove one dim of tensor of logits

            pred_probs.append(pred_prob)   
    # turn list into a tensor
    return torch.stack(pred_probs)



if __name__ == "__main__":

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--test_path", type=str, required=True, help="Path to the .pt test dataset")
        args = parser.parse_args()
    except Exception as e:
        print(f"Error parsing arguments: {e}")
        exit(1)
    
    # Load training data to get class names
    try:
        data_dir = "data"
        train_data_path = os.path.join(data_dir, 'fashion_mnist_train.pt')
        if not os.path.exists(train_data_path):
            raise FileNotFoundError(f"Training data file not found at {train_data_path}")
        train_data = torch.load(train_data_path)
        class_names = train_data.classes
    except Exception as e:
        print(f"Error loading training data: {e}")
        exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Device is {device}")
    
    # Load the saved model
    try:
        model_path = "cnn_model.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        loaded_model = torch.load(model_path, map_location=device)
        loaded_model.to(device)
        loaded_model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # Load the test data
    try:
        if not os.path.exists(args.test_path):
            raise FileNotFoundError(f"Test dataset file not found at {args.test_path}")
        test_data = torch.load(args.test_path)
    except Exception as e:
        print(f"Error loading test data: {e}")
        exit(1)

    # Make predictions
    try:
        test_samples = []
        test_labels = []
        for sample, label in random.sample(list(test_data), k=6):  # Sample 6 data points from test data
            test_samples.append(sample)
            test_labels.append(label)

        gt_class_names = [class_names[test_labels[i]] for i in range(len(test_samples))]
        print("\nTrue label names:\n\n", gt_class_names)
        pred_probs = make_predictions(model=loaded_model, data=test_samples, device=device)
        pred_classes = pred_probs.argmax(dim=1)
        pred_class_names = [class_names[pred_classes[i]] for i in range(len(test_samples))]
        print("\nPredicted label names:\n\n", pred_class_names)
    except Exception as e:
        print(f"Error during prediction: {e}")
        exit(1)