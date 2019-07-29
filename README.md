# pytorch-unet

PyTorch implementation of [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (Ronneberger et al., 2015).
This implementation has many tweakable options such as:
- Depth of the network
- Number of filters per layer
- Transposed convolutions vs. bilinear upsampling
- valid convolutions vs padding
- batch normalization

# Documentation
```python
class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False,
                 batch_norm=False, up_mode='upconv'):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
```
An example of how to use the network
```python
import torch
import torch.nn.functional as F
from unet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_classes=2, padding=True, up_mode='upsample').to(device)
optim = torch.optim.Adam(model.parameters())
dataloader = ...
epochs = 10

for _ in range(epochs):
    for X, y in dataloader:
        X = X.to(device)  # [N, 1, H, W]
        y = y.to(device)  # [N, H, W] with class indices (0, 1)
        prediction = model(X)  # [N, 2, H, W]
        loss = F.cross_entropy(prediction, y)

        optim.zero_grad()
        loss.backward()
        optim.step()
```

# Discussion of parameters/architecture
Some of the architecture choices in other implementations (i.e. 'same' padding) differ from the original implementation. Unfortunately, the paper doesn't really go into detail on some these choices. But in practice, they can be quite important. Here I will discuss some settings and provide a recommendation for picking them.

### SAME vs VALID padding
The original paper uses VALID padding (i.e. no padding), so the height and width of the feature map decreases after each convolution. Most implementations found online use SAME padding (i.e. zero padding by 1 on each side) so the height and width of the feature map will stay the same (not completely true, see "_Input size_" below). The main benefit of using SAME padding is that the output feature map will have the same spatial dimensions as the input feature map. In the original paper, the output feature map is smaller. So if you want your output to be of a certain size, you have to do (a lot of) padding on the input image.

Although using VALID padding seems a bit more inconvenient, I would still recommend using it. When using SAME padding, the border is *polluted* by zeros in each conv layer. Resulting in a *border-effect* in the final output. For instance, a lot of pixels won't have had enough information as input, so their predictions are not as accurate. When using VALID padding, each output pixel will only have seen "real" input pixels.

### Upsampling vs Transposed convolutions
The original paper uses transposed convolutions (a.k.a. upconvolutions, a.k.a. fractionally-strided convolutions, a.k.a deconvolutions) in the "up" pathway. Other implementations use (bilinear) upsampling, possibly followed by a 1x1 convolution. The benefit of using upsampling is that it has no parameters and if you include the 1x1 convolution, it will still have less parameters than the transposed convolution. The downside is that it can't use weights to combine the spatial information in a smart way, so transposed convolutions can potentially handle more fine-grained detail.

I would recommend to use upsampling by default, unless you know that your problem requires high spatial resolution. Still, you can easily experiment with both by just changing the `up_mode` parameter.

### Input size
When running the model on your own data, it is important to think about what size your input (and output) images are. Although this is more straightforward when using `padding=True` (i.e., SAME), the output size is not always equal to your input size. In particular, your input size needs to be `depth - 1` times divisible by 2. The reason is that max-pool layers will divide their input size by 2, rounding down in the case of an odd number. For instance, when your input has `width = height = 155`, and your U-net has `depth = 4`, the output of each block will be as follows:

```
[  Downsampling  ]    [   Upsampling   ]
155 -> 72 -> 36 -> 18 -> 36 -> 72 -> 144
```
If your labels are 155x155, you will get a mismatch in the size between your predictions and labels. The solution is to pad your input with zeros (for instance using `np.pad`). In this example, you could pad your input to 160x160 (which is 3 times divisible by 2), and then crop your labels before computing the loss. An alternative is to center-crop your labels to match the size of the predictions. In that case you don't have to pad with zeros.
