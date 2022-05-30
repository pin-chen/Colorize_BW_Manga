import glob
from PIL import Image
for x in glob.glob('Picture\oringinal\*'):
    img = Image.open(x)
    y = x.split('\\')
    imgGray = img.convert('L')
    imgGray.save(y[0]+'/grayscale/'+y[2])