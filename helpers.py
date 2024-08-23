from pathlib import Path
from tokenizers import Tokenizer
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from config import getConfig, get_hyperparams
import pandas as pd
from torchvision import transforms
from PIL import Image
import torch

config = getConfig()

def get_tokeniser(ds=None):

    def get_all_sentences(ids):
        for text in ids["caption"]:
            if type(text) == float:
                continue
            else:
                yield text

    tokenizer_path = Path(config["tokenizer_file"])

    if tokenizer_path.exists():

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer

    else:

        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds), trainer=trainer)
        tokenizer.save(str(tokenizer_path))

        return tokenizer
    
def get_dataframe_preprocessed(path : str) -> pd.DataFrame:
    results = pd.read_excel(path)    
    mask = results[results.columns[1]].apply(lambda x: not isinstance(x,float))
    results.columns = ["image","caption"]
    results = results[mask]
    results[results.columns[1]]=results[results.columns[1]].apply(lambda x: x.strip())
    results = results.iloc[1:,:]
    return results

def get_transforms():

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return transform

def getPatches(image):
    transforms = get_transforms()
    encoder_input = transforms(image)

    # Ensure the image tensor is in the correct shape (C, H, W)
    encoder_input = torch.tensor(encoder_input, dtype=torch.float32)

    if encoder_input.size(0) != 3:
        raise ValueError(f"Expected 3 channels, but got {encoder_input.size(0)}")

    # Extract patches
    hp = get_hyperparams()

    patches = encoder_input.unfold(1, hp["patch_size"], hp["stride"]).unfold(2, hp["patch_size"], hp["stride"])
    patches = patches.contiguous().view(3, -1, hp["patch_size"], hp["patch_size"])
    patches = patches.permute(1, 0, 2, 3)  # Shape: (num_patches, 3, patch_size, patch_size)
    num_patches = patches.shape[0]
    patches = patches.reshape(num_patches, -1).unsqueeze(0)  # Shape: (num_patches, C * patch_size * patch_size)

    return patches