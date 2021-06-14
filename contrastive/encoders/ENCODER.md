# Encoders

Here are the encoders that apply different contrastive learning frameworks.  
So far there are four types of frameworks available: 
[SimCLR](https://arxiv.org/abs/2002.05709), 
[MoCo](https://arxiv.org/abs/1911.05722), 
[BYOL](https://arxiv.org/abs/2006.07733),
[SimSiam](https://arxiv.org/abs/2011.10566).  
One can utilize `get_encoder()` in [`utils.py`](https://github.com/Kaminyou/Template-is-all-you-need/blob/main/utils.py) 
to get the desired encoder. 
For more details please look at `get_encoder()`'s docstring.  

The `forward` function of the encoder looks like this:  
```python
def forward(self, img1, img2=None, return_loss=False)
```
If `return_loss` is set to `True`, the encoder will return the embeddings 
of the two images and their contrastive loss.  
If `return_loss` is set to `False`, the encoder will return the embedding
of the input image.

Here is a small example of how to use the encoder in your code:  
```python
from utils import get_encoder

# instantiate an encoder
encoder = get_encoder('resnet50', 'simclr')

# if return_loss = True, the encoder will return the embeddings 
# of img1 and img2 as well as the contrastive loss
code1, code2, contrastive_loss = encoder(img1, img2, return_loss=True)

# if return_loss = False (default), the encoder will return the embedding
# of the input image
code = encoder(img, return_loss=False) # or
code = encoder(img)
```
