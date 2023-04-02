#imports are done here

import sensor, image, lcd, time #imports are done here
import KPU as kpu

###############################
#settings for the screen below#
###############################

lcd.init()
lcd.rotation(2)
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((224, 224)) #changes based on the trained model
sensor.set_hmirror(1)
sensor.set_vflip(0) #flips camera
sensor.run(1)
lcd.clear()

lcd.draw_string(0,96,"Imports Are Done")

#Loads the model from the memory
task = kpu.load(0x300000)

#Creates the labels based on the trained model
labels = ['withMask', 'withoutMask']

#Manually sets the shape of the output layer
outputs = kpu.set_outputs(task, 0, 1, 1, 2)


lcd.draw_string(0,112,"Preparations are done")

#Detection starts
while True:
    kpu.memtest()
    img = sensor.snapshot()
    feature_map = kpu.forward(task, img)
    plist = feature_map[:]
    pmax = max(plist)
    max_index = plist.index(pmax)

    # Draw the detected class label on the image
    if labels[max_index].strip() == 'withMask':
        img.draw_string(0, 195, 'withMask', color=(0, 255, 0), scale=3)
    else:
        img.draw_string(0, 195, 'withoutMask', color=(255, 0, 0), scale=3)

    val1 = lcd.display(img)

val1 = kpu.deinit(task)
