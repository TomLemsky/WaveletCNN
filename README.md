# PyTorch WaveletCNN

My PyTorch implementation of the WaveletCNN architecture by Fujieda et al. ( https://arxiv.org/abs/1707.07394 )

Details were based on the Caffe implmentation: https://github.com/shinfj/WaveletCNN_for_TextureClassification

## Requirements

- Python3
- PyTorch
- pytorch_wavelets

## Usage

Import like this:

```
import wavelet_cnn
model = WaveletCNN(in_channels=3, num_classes=10, init_weights=True, prelayers=False)
```

and then use the model as you would any other PyTorch model. 
Softmax or Logits are not included, must use a loss function that includes them (e.g. 
torch.nn.BCEWithLogitsLoss) or apply them during training.
