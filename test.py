import pyautogui
import threading, time
import keyboard
from pynput.keyboard import Key, Controller
import win32api, win32con

# 58px per tile
# 231, 71, 29 = RED
# (670, 314) inclusive
# to (1249, 835) inclusive
# region=(670,314,580,522)

FRAMERATE = 0.135

def click(x, y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)

def press(key):
    keyboard = Controller()
    keyboard.press(key)
    keyboard.release(key)

def scan():
    frame_elapsed = 0

    # click(860, 650)
    # press('d')

    pic = pyautogui.screenshot(region=(670,314,580,522))
    x, y = 0, 0

    for i in range(0, 550, 58):
        for j in range(0, 500, 58):
            y = y % 9 
            r,g,b = pic.getpixel((25+i, 25+j))
            if r == 231:
                print(f"at {x,y} color is {r,g,b} at time {time.time()} and frame {frame_elapsed}")
            y += 1
        x += 1

    frame_elapsed += 1

    threading.Timer(5, scan).start()

scan()
    
# check screen for apple
"""while True:
    if pyautogui.locateOnScreen(r"apple.png") != None:
        print("I can see it")
        time.sleep(0.5)
    else: 
        print("i cant see it")
        time.sleep(0.5)
        

    time.sleep(1)
    pyautogui.useImageNotFoundException()
    try:
        location = pyautogui.locateOnScreen('apple.png', region=(670,314,580,522), grayscale=True, confidence=0.8)
        print('image found')
    except pyautogui.ImageNotFoundException:
        print('ImageNotFoundException: image not found')"""



# check for position of mouse on screen
"""while True:
    pyautogui.displayMousePosition()

    im = pyautogui.screenshot()
    for i in range(800):
        print(im.getpixel((700, 200 + i)), "y=", 200+i)"""



# open file w/ multiple numbers and read from it 
"""with open("epoch.txt") as f:
    for line in f:
        numbers_str = line.split()
        numbers_float = [float(x) for x in numbers_str]
        print(numbers_float[1])"""