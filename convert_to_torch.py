import torch
import os
from torchvggish.vggish import VGGish
os.environ['KMP_DUPLICATE_LIB_OK']='True'
model = VGGish("1")
# model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()

# Download an example audio file
# import urllib
# url, filename = ("http://soundbible.com/grab.php?id=1698&type=wav", "bus_chatter.wav")
# try: urllib.URLopener().retrieve(url, filename)
# except: urllib.request.urlretrieve(url, filename)
filename = "1001_DFA_ANG_XX.wav"
test = model.forward(filename,overlaprate=0.9)
test