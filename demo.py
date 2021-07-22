import pydicom
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

demo1 = {
    'path': 'demo/1.dcm',
    'boxes': [{'x': 789.28836, 'y': 582.43035, 'width': 1026.65662, 'height': 1917.30292}, {'x': 2245.91208, 'y': 591.20528, 'width': 1094.66162, 'height': 1761.54944}]
}
demo2 = {
    'path': 'demo/2.dcm',
    'boxes': [{'x': 677.42216, 'y': 197.97662, 'width': 867.79767, 'height': 999.78214}, {'x': 1792.69064, 'y': 402.5525, 'width': 617.02734, 'height': 1204.358}]
}


def show(demo):
    path = demo['path']
    ds = pydicom.dcmread(path)
    boxes = demo['boxes']
    plt.figure()
    for i in range(len(boxes)):
        x, y = boxes[i]['x'], boxes[i]['y']
        width, height = boxes[i]['width'], boxes[i]['height']
        current_axis = plt.gca()
        current_axis.add_patch(Rectangle((x - .1, y - .1), width, height, fill=None, alpha=1))
    plt.imshow(ds.pixel_array)
    plt.show()


if __name__ == '__main__':
    show(demo2)
