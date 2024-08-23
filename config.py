def getConfig():

    return {
        "tokenizer_file" : 'tokeniser_0.json',
        "batchSize" : 8
    }

def get_hyperparams(tokenizer=None):

    d_model = 512
    num_heads = 8
    num_layers = 6
    vocab_size = 13421
    if(tokenizer is not None):
        vocab_size = tokenizer.get_vocab_size()
    max_seq_len = 128
    patch_size = 32
    stride = 32
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 10
    model_save_name = "final_2.pth"

    parms = {
        "num_patches" : (256 // patch_size) ** 2, 
        "patch_dim" : 3 * patch_size * patch_size,  
        "d_model" : d_model,
        "num_heads": num_heads,
        "num_layers" : num_layers,
        "vocab_size" : vocab_size,
        "max_seq_len" : max_seq_len,
        "stride" : stride,
        "batch_size" : batch_size,
        "learning_rate" : learning_rate,
        "num_epochs" : num_epochs,
        "patch_size" : patch_size,
        "save_name" : model_save_name
    }

    return parms