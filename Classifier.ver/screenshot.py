from time import sleep
import pyautogui
import os


def screen_shot():
    if not os.path.exists('Picture'):
        os.mkdir('Picture')
    for i in range(600):
        myScreenshot = pyautogui.screenshot()
        myScreenshot.save(f'Picture/screenshot{i}.png')
        sleep(2)


if __name__ == '__main__':
    screen_shot()
