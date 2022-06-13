import pyautogui
from time import sleep

def screen_shot():
    for i in range(600):
        myScreenshot = pyautogui.screenshot()
        myScreenshot.save(f'Pictures/oringinal/{i}.png')
        sleep(2)

if __name__ == '__main__':
    screen_shot()
