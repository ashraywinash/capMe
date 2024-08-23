
import torch
import torchvision
from torch.utils.data import Dataset
import pandas as pd

from PIL import Image
from tokenizers import Tokenizer



class ImageCaptionDataset(Dataset):


    def __init__(self, ds: pd.DataFrame, transforms: torchvision.transforms, tokenizer_tgt : Tokenizer, seq_len = 128, patch_size  = 32, stride = 32) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.ds = ds
        self.transforms = transforms
        self.patch_size = patch_size
        self.stride = stride
        self.tokenizer_tgt = tokenizer_tgt
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    
    def __getitem__(self, index):
        # Image reading and transforms
        image_path = "/Users/ashu/CODE/Python_Programming/deepLearning/ImageCaptioning/flickr8k_images/flickr8k_images/" + self.ds.iloc[index, :]["image"]
        # image_path = "/Users/ashu/CODE/Python_Programming/deepLearning/capme/image.png"
        print(image_path)
        image = Image.open(image_path).convert('RGB')  # Convert to RGB
        encoder_input = self.transforms(image)

        # Ensure the image tensor is in the correct shape (C, H, W)
        encoder_input = torch.tensor(encoder_input, dtype=torch.float32)

        if encoder_input.size(0) != 3:
            raise ValueError(f"Expected 3 channels, but got {encoder_input.size(0)}")

        # Extract patches
        patches = encoder_input.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
        patches = patches.contiguous().view(3, -1, self.patch_size, self.patch_size)
        patches = patches.permute(1, 0, 2, 3)  # Shape: (num_patches, 3, patch_size, patch_size)
        num_patches = patches.shape[0]
        patches = patches.reshape(num_patches, -1)  # Shape: (num_patches, C * patch_size * patch_size)

        # Text processing
        tgt_text = self.ds.iloc[index, :]["caption"]
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Ensure proper padding
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * max(dec_num_padding_tokens, 0), dtype=torch.int64),
        ], dim=0)

        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * max(dec_num_padding_tokens, 0), dtype=torch.int64),
        ], dim=0)

        # Ensure correct sequence length
        decoder_input = torch.cat([decoder_input, torch.tensor([self.pad_token] * (self.seq_len - len(decoder_input)), dtype=torch.int64)])
        label = torch.cat([label, torch.tensor([self.pad_token] * (self.seq_len - len(label)), dtype=torch.int64)])

        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # Return patches if using Vision Transformer or similar architecture
        return patches, decoder_input, label