
import errno
import glob


def GetCoordinate():
    coordinate = []
    path = './label_2/*.txt'
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
            return coordinate
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                print("error occurred here")
                raise

def createBoxes(images, coordinates):
    seq_length = len(coordinates)
    for n in range(seq_length):
        frame = seq_length[n]
        for i in range(len(frame)):
            car = frame[i]
            pixel_start_x = int(round(car['x1']))
            pixel_start_y = int(round(car['y1']))
            pixel_end_x = int(round(car['x2']))
            pixel_end_y = int(round(car['y2']))
            for ri in range(pixel_end_y-pixel_start_y+1):
                for ci in range(pixel_end_x-pixel_start_x+1):
                    images[pixel_start_y+ri, pixel_start_x+ci, n] = 1
    return images


def main():
    cc = GetCoordinate()
    print(len(cc))

main()
