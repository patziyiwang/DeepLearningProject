
import os




def GetCoordinate():
    directory = './label_2'
    coordinate = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            cars = []
            with open('./label_2/000002.txt') as o:
                for i, t in enumerate(o.readlines(), 1):
                    l_s = t.split()
                    dic = {}
                    if l_s[0] == 'Car':
                        dic['x1'] = l_s[4]
                        dic['y1'] = l_s[5]
                        dic['x2'] = l_s[6]
                        dic['y2'] = l_s[7]
                        cars.append(dic)
        coordinate.append(cars)
        





def createBoxes(images, coordinates):
    seq_length = images.shape[2]
    for n in range(seq_length):
        pixel_start_x = int(round(x1[n]))
        pixel_start_y = int(round(y1[n]))
        pixel_end_x = int(round(x2[n]))
        pixel_end_y = int(round(y2[n]))
        for ri in range(pixel_end_y-pixel_start_y+1):
            for ci in range(pixel_end_x-pixel_start_x+1):
                images[pixel_start_y+ri, pixel_start_x+ci, n] = 1
    return images

