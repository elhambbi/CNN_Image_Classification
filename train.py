import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm   
from timeit import default_timer as timer 
import os


# CNN
class cnnmodel1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()         #inherit from nn.Module
        self.conv_block_1 = nn.Sequential(          
        nn.Conv2d(in_channels=input_shape,       #input layer
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),                                #activation layer                       
        nn.Conv2d(in_channels=hidden_units,       #convolutional layer
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),                                #activation layer   
        nn.MaxPool2d(kernel_size=2)               #pooling layer
        )
        self.conv_block_2 = nn.Sequential(            #convolutional layer
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),                               #activation layer  
        nn.Conv2d(in_channels=hidden_units,      #convolutional layer
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),                               #activation layer  
        nn.MaxPool2d(kernel_size=2)              #pooling layer
       )
        self.classifier = nn.Sequential(         #output layer
        nn.Flatten(),                               #flatten the output of the two blocks into a single feature vector
        nn.Linear(in_features=hidden_units*7*7,     # 10*7*7 is the output shape from conv_block2
                  out_features=output_shape)
        )
    
    def forward(self, x):
        x = self.conv_block_1(x)
        #print(f"Output shape of conv_block_1: {x.shape}")
        x = self.conv_block_2(x) 
        #print(f"Output shape of conv_block_2: {x.shape}")
        x = self.classifier(x)
        #print(f"Output shape of classifier: {x.shape}")
        return x
    

# loss function, evaluation metric and optimizer
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


# TRAIN
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device):
    
    train_loss, train_acc = 0, 0

    # Put model into training mode
    model.train()

    # Add a loop to loop through the training batches
    for batch, (X, y) in enumerate(data_loader):
        # Put data on target device 
        X, y = X.to(device), y.to(device)

        # 1. Forward pass (outputs the raw logits from the model)
        y_pred = model(X)

        # 2. Calculate loss and accuracy (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulate train loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=torch.softmax(y_pred.squeeze(), dim=1).argmax(dim=1)) # go from logits -> prediction labels

        # 3. Optimizer zero grad. Sets the gradients of all optimized torch.Tensors to zero.
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step (update the model's parameters once per batch)
        optimizer.step()

    # Divide total train loss and acc by length of train dataloader (getting the average for all batches)
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")


# TEST
def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device):
    
    test_loss, test_acc = 0, 0
  
    # Put the model in eval mode
    model.eval()

    # Turn on inference mode context manager
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            # 1. Forward pass (outputs raw logits)
            test_pred = model(X)
           
            # 2. Calculuate the loss/acc
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                                  y_pred=torch.softmax(test_pred.squeeze(), dim=1).argmax(dim=1)) # go from logits -> prediction labels 
            
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")



# a function to calculate training time
def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")


# model evaluation
torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, 
               accuracy_fn,
               device):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            X, y = X.to(device), y.to(device)
            # Make predictions
            y_pred = model(X)

            # Accumulate the loss and acc values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                             y_pred=torch.softmax(y_pred.squeeze(), dim=1).argmax(dim=1))  # go from logits -> prediction labels 

        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
    
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
          "model_loss": loss.item(),
          "model_acc": acc}




if __name__ == "__main__":

    # load the saved training and test datasets
    data_dir = "data"
    try:
        train_data_path = os.path.join(data_dir, 'fashion_mnist_train.pt')
        if not os.path.exists(train_data_path):
            raise FileNotFoundError(f"Training data file not found at {train_data_path}")
        train_data = torch.load(train_data_path)
        class_names = train_data.classes
        class_to_idx = train_data.class_to_idx
    except Exception as e:
        print(f"Error with training dataset: {e}")
        exit(1)

    try:
        test_data_path = os.path.join(data_dir, 'fashion_mnist_test.pt')
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test data file not found at {test_data_path}")
        test_data = torch.load(test_data_path)
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        exit(1)

    # create Dataloader
    BATCH_SIZE = 32
    # turns our dataset which is in the form of PyTorch Datasets into a Python iterable (batches)
    train_dataloader = DataLoader(dataset= train_data,
                                batch_size= BATCH_SIZE,
                                shuffle= True)
    test_dataloader = DataLoader(dataset= test_data,
                                batch_size= BATCH_SIZE,
                                shuffle= False)
    print(f"Length of train_dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
    print(f"Length of test_dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")

    # cpu or gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device is {device}")

    #creating the model with 10 neurons for hidden layers
    torch.manual_seed(42)
    model1 = cnnmodel1(input_shape=1,       #if the images were colored, input_shape=3 for RGB
                    hidden_units=10, 
                    output_shape=len(class_names)).to(device)
    
    loss_fn = nn.CrossEntropyLoss()   # multiclass classification
    optimizer = torch.optim.SGD(params=model1.parameters(),lr=0.1)

    # Train and test the model
    torch.manual_seed(42)
    start_time= timer()
    epochs = 5
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n")
        train_step(model=model1,
                data_loader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                accuracy_fn=accuracy_fn,
                device=device)
        test_step(model=model1,
                data_loader=test_dataloader,  # was better to use an validation dataset here
                loss_fn=loss_fn,
                accuracy_fn=accuracy_fn,
                device=device)

    end_time = timer()
    print_train_time(start=start_time, end=end_time,
                    device=str(next(model1.parameters()).device))

    # save the last model after all epoches
    try:
        torch.save(model1, "cnn_model.pt")
        # torch.save(model1.state_dict(), "cnn_model_weights.pth")  # to only save model parameters (not architecture)
        print("Final model saved")
    except Exception as e:
        print(f"Error saving model: {e} ")
        exit(1)

    # results
    results = eval_model(
        model=model1,
        data_loader=test_dataloader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device
    )
    print("Trained model results:\n", results)