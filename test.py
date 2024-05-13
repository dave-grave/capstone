import pyautogui
import time
import keyboard

# 58px per tile
# 231, 71, 29 = RED
# (670, 314) inclusive
# to (1249, 835) inclusive
# region=(670,314,580,522)

while keyboard.is_pressed('q') == False:

    pic = pyautogui.screenshot(region=(670,314,580,522))

    width, height = pic.size
    x, y = 0, 0

    for i in range(0, 550, 58):
        
        for j in range(0, 500, 58):
            y = y % 9 
            r,g,b = pic.getpixel((25+i, 25+j))
            if r == 231:
                print(f"at {x,y} color is {r,g,b}")
                
            y += 1
        x += 1



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