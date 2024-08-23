import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from config import getConfig, get_hyperparams
from helpers import get_tokeniser, get_dataframe_preprocessed, get_transforms
from dataset import ImageCaptionDataset
from model import TransformerImageCaptioning
from tqdm import tqdm
from tokenizers import Tokenizer
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

# preload model


def train_model(model, 
                dataloader,
                optimizer,
                criterion,
                num_epochs,
                vocab_size,
                mask,
                file_path):


    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        # Wrap the dataloader with tqdm for progress bar
        for patches, decoder_input, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            
            patches = patches.to(device)
            decoder_input = decoder_input.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            # Forward pass
            outputs = model(patches, decoder_input, mask)
            # Compute loss
            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
            epoch_loss += loss.item()
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        
        
        # Print the average loss for the epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")
        # Save the model state_dict
        torch.save(model.state_dict(), file_path)


        with open("logs.txt", 'a') as f:  
            f.write(f"\nEpoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}\n")


if __name__ == '__main__':
    device : str = "cpu"
    torch.device = device
    config : dict = getConfig()
    # read the df and preprocess it
    results : pd.DataFrame = get_dataframe_preprocessed('./captions.xls')
    # tokeniser
    tokenizer : Tokenizer = get_tokeniser(results)
    # Image transforms
    transform : transforms = get_transforms()

    # Define hyperparameters
    hp = get_hyperparams(tokenizer)

    # Load the dataset and make it into batches
    dataset = ImageCaptionDataset(results, transform, tokenizer, seq_len=hp["max_seq_len"], patch_size=hp["patch_size"], stride=hp["stride"])
    dataloader = DataLoader(dataset, batch_size=hp["batch_size"], shuffle=True, num_workers=4)

    model_path = Path("models/" + hp["save_name"])
    print("Model Path Exists:", model_path.exists())

    if model_path.exists():    
        model = TransformerImageCaptioning(**hp).to(device)
        model.load_state_dict(torch.load('models/' + hp["save_name"]))
    else:
        model = TransformerImageCaptioning(**hp).to(device)

    # Training ....
    # loss function, and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_token.item())
    optimizer = optim.Adam(model.parameters(), lr=hp["learning_rate"])

    mask = nn.Transformer.generate_square_subsequent_mask(hp["max_seq_len"])

    train_model(model, dataloader, optimizer, criterion, hp["num_epochs"], tokenizer.get_vocab_size(), mask, "models/" + hp["save_name"])

