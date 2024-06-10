from pynput.mouse import Controller, Button
import time
time.sleep(5)
mouse = Controller()
import keyboard
while True:
    if keyboard.is_pressed('j'):
            while True:
                mouse.click(Button.right)
                time.sleep(0.1)
                if keyboard.is_pressed('k'):
                    break