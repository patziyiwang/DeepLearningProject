
import errno
import glob
coordinate = []
    path = './label_2/*.txt'
    files = glob.glob(path)
    for name in files:
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

