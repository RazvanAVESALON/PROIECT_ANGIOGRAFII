import imageio
import pathlib
filenames=r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\Experiment_Dice_index11282022_1227\Test12222022_1347\OVERLAP_Colored_0d2e685e8a404667b62dd47cc7b728c3_92408191-1.png"
path=r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\Experiment_Dice_index11282022_1227\Test12222022_1347\OVERLAP_Colored_0d2e685e8a404667b62dd47cc7b728c3_92408191-2.png"
images = []

images.append(imageio.imread(filenames))
images.append(imageio.imread(path))
cale=pathlib.Path.cwd()
kargs = { 'duration': 1}

imageio.mimsave(exportname, images,'GIF',**kargs )