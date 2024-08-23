import torch
import pandas as pd
from dataset import ImageCaptionDataset
from model import TransformerImageCaptioning
from config import get_hyperparams
from helpers import get_dataframe_preprocessed, get_tokeniser, get_transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random

import warnings
warnings.filterwarnings('ignore')

def top_k_sampling(logits, k=2):
    # Apply top-k filtering to the logits
    top_k_logits, top_k_indices = torch.topk(logits, k)
    top_k_logits = top_k_logits - torch.max(top_k_logits)  # for numerical stability
    probs = F.softmax(top_k_logits, dim=-1)
    
    # Sample from the top-k logits
    next_token = torch.multinomial(probs, 1).squeeze()

    return top_k_indices[:,next_token]



def generate_caption(model, patches, start_token_idx, eos_token_idx, max_caption_length, device, tokenizer):
    model.eval()  
    batch_size = patches.shape[0]
    # Initialize the decoder input with the start token
    decoder_input = torch.full((batch_size, 1), start_token_idx, dtype=torch.int64, device=device)
    # Move the input data to the appropriate device
    patches = patches.to(device)
    # Initialize an empty list to store the generated caption tokens
    caption_tokens = []
    for _ in range(max_caption_length):
        with torch.no_grad():
            
            seq_len = decoder_input.size(1)
            # The mask is True where we want to block attention (future positions)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            outputs = model(patches, decoder_input, mask)
            # next_word_token = outputs.argmax(-1)[:, -1]
            next_word_token = top_k_sampling(outputs[:,-1,:])
            caption_tokens.append(next_word_token.item())
            
            # If the end-of-sequence token is generated, stop
            if next_word_token.item() == eos_token_idx:
                break
            
            # Update the decoder input by appending the predicted word
            decoder_input = torch.cat([decoder_input, next_word_token.unsqueeze(1)], dim=1)

    caption = [tokenizer.id_to_token(i) for i in caption_tokens]
    caption = ' '.join(caption[0:-1])
    
    return caption


if __name__ == '__main__':
    device = "cpu"
    # read the df and preprocess it
    results : pd.DataFrame = get_dataframe_preprocessed('./captions.xls')
    tokenizer = get_tokeniser(results)

    hp = get_hyperparams(tokenizer)
    transform = get_transforms()
    dataset = ImageCaptionDataset(results, transform, tokenizer, seq_len=hp["max_seq_len"], patch_size=hp["patch_size"], stride=hp["stride"])
    dataloader = DataLoader(dataset, batch_size=hp["batch_size"], shuffle=True, num_workers=4)

    model = TransformerImageCaptioning(**hp).to(device)
    model.load_state_dict(torch.load('models/final_2.pth'))
    ran = random.randint(0,40000)
    patches, decoder_input, label = dataset.__getitem__(ran)

    print(f"Expected : {' '.join([tokenizer.id_to_token(i) for i in decoder_input if i not in [0,1,3,2]])}")

    patches = patches.unsqueeze(0)

    caption = generate_caption(model,patches,2,3,50,device,tokenizer)
    print(f"Actual : {caption}")
