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



if __name__== "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, help="Path to the .pt test dataset")
    args = parser.parse_args()

    # class_names are fixed - or just define class_names manually
    data_dir = "data"
    train_data = torch.load(os.path.join(data_dir, 'fashion_mnist_train.pt'))
    class_names = train_data.classes

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device is {device}")

    # load the saved model
    loaded_model = torch.load("cnn_model.pt", map_location= device)
    # loaded_model = cnnmodel1()   # if only the parameters (weights) are saved
    # loaded_model.load_state_dict(torch.load("cnn_model_weights.pth"))  
    loaded_model.to(device) 
    loaded_model.eval()

    # a sample of size 6 from test data 
    # random.seed(42)
    test_data = torch.load(args.test_path)
    test_samples = [] 
    test_labels = []
    for sample, label in random.sample(list(test_data), k=6):   # test_data is not a dataloader
        test_samples.append(sample)
        test_labels.append(label)
    gt_class_names = [class_names[test_labels[i]] for i in range(len(test_samples))]
    print("\nTrue label names:\n\n", gt_class_names)
    # make predictions
    pred_probs = make_predictions(model= loaded_model,
                                data= test_samples,
                                device= device)
    # convert prediction probabilities to labels
    pred_classes = pred_probs.argmax(dim=1)
    pred_class_names = [class_names[pred_classes[i]] for i in range(len(test_samples))]
    print("\nPredicted label names:\n\n", pred_class_names)