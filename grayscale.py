import glob
from PIL import Image

def grayscale():
    for x in glob.glob('Pictures\oringinal\*.png'):
        img = Image.open(x)
        y = x.split('\\')
        imgGray = img.convert('L')
        imgGray.save(y[0]+'/grayscale/'+y[2])
    
if __name__ == '__main__':
    grayscale()