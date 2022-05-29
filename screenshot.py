import pyautogui
from time import sleep


def screen_shot():
    for i in range(6000):
        myScreenshot = pyautogui.screenshot()
        myScreenshot.save(f'./Picture/test{i}.png')
        sleep(2)
        


if __name__ == '__main__':
    screen_shot()
