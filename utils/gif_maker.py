import imageio
import pathlib
import numpy as np 
path=r"E:\__RCA_bif_detection\data\00cca518a10d41adb9476aefc38a0b69\40117765\frame_extractor_frames.npz"
images = []

img=np.load(path)['arr_0']
print (img.shape)
for frame in range(14) :
    images.append(img[frame])
cale=pathlib.Path.cwd()
kargs = { 'duration': 1}

imageio.mimsave("ex.gif", images,'GIF',**kargs )