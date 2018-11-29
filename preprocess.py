import numpy as np
import errno
import glob
import numpy as np
import pickle

def GetCoordinate(data_dir):
    coordinate = []
    path = data_dir + '*.txt'
    files = glob.glob(path)
    for name in files:
        try:
            with open(name) as f:
                cars = []
                for i, t in enumerate(f.readlines(), 1):
                    l_s = t.split()
                    dic = {}
                    if l_s[0] == 'Car':
                        dic['x1'] = l_s[4]
                        dic['y1'] = l_s[5]
                        dic['x2'] = l_s[6]
                        dic['y2'] = l_s[7]
                        cars.append(dic)
            coordinate.append(cars)

        except IOError as exc:
            if exc.errno != errno.EISDIR:
                print("error occurred here")
                raise
    return coordinate

def createBoxes(images, coordinates):
    seq_length = len(coordinates)
    for n in range(seq_length):
        frame = coordinates[n]
        for i in range(len(frame)):
            car = frame[i]
            pixel_start_x = int(round(float(car['x1'])))
            pixel_start_y = int(round(float(car['y1'])))
            pixel_end_x = int(round(float(car['x2'])))
            pixel_end_y = int(round(float(car['y2'])))
            for ri in range(pixel_end_y-pixel_start_y+1):
                for ci in range(pixel_end_x-pixel_start_x+1):
                    images[pixel_start_y+ri, pixel_start_x+ci, n] = 1
    return images

def saveImages(images, name):
    pickle.dump(images, open(name + ".pkl", "wb"))

def main():
    cc = GetCoordinate('./label_2/')
    print(cc)
    images = np.zeros((512,1392,7481))
    new_images = createBoxes(images, cc)
    saveImages(new_images, 'bbox_data')

main()