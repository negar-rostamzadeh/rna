import cPickle
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2


seed = 1234
rng = np.random.RandomState(seed)
clutter_shape = np.array([7, 7])
num_clutters = 8
num_circles = 16
time_steps = 10
max_velocity = 10


with open("/data/lisa/data/mnist/mnist.pkl", "rb") as f:
    dataset = cPickle.load(f)
    digits, np_targets = [
        np.concatenate(xs, axis=0) for xs in zip(*dataset)]
    dataset = None
    digits = digits.reshape((digits.shape[0], 28, 28))

# 100 different clutters
i = rng.randint(digits.shape[0], size=(100,))
# don't take too close to edge to avoid empty patches
margin = 2
maxtopleft = (np.array(digits.shape[1:]) -
              2 * margin - clutter_shape)
# randint doesn't support high=some_shape, so use random_sample
topleft = np.round(
    rng.random_sample((100, 2)) * maxtopleft).astype(int)
y = (margin + topleft[:, 0, np.newaxis] +
     np.arange(clutter_shape[0])[np.newaxis, ...])
x = (margin + topleft[:, 1, np.newaxis] +
     np.arange(clutter_shape[1])[np.newaxis, ...])

all_clutters = digits[i[:, np.newaxis, np.newaxis],
                      y[:, :, np.newaxis],
                      x[:, np.newaxis, :]]

all_frames = []
all_locations = []
t0 = time.time()
for exp in range(70000):
    if exp % 1000 == 0:
        print exp
    digit_all_locations = []
    digit = digits[exp]
    digit_location_init = [14 + rng.randint(72), 14 + rng.randint(72)]
    sign_x = 2 * rng.randint(2) - 1
    sign_y = 2 * rng.randint(2) - 1
    x_velocity = sign_x * rng.randint(11)
    digit_velocity = [
        x_velocity,
        sign_y * int(np.sqrt(max_velocity ** 2 - x_velocity ** 2))]

    clutters = []
    clutters_locations_init = []
    clutters_velocities = []
    for i in range(num_clutters):
        clutter = all_clutters[rng.randint(91)]
        clutter_location_init = [3 + rng.randint(94), 3 + rng.randint(94)]
        sign_x = 2 * rng.randint(2) - 1
        sign_y = 2 * rng.randint(2) - 1
        x_velocity = sign_x * rng.randint(11)
        clutter_velocity = [
            x_velocity,
            sign_y * int(np.sqrt(max_velocity ** 2 - x_velocity ** 2))]

        clutters.append(clutter)
        clutters_locations_init.append(clutter_location_init)
        clutters_velocities.append(clutter_velocity)

    circles = []
    circles_locations_init = []
    circles_velocities = []
    for i in range(num_circles):
        circle = cv2.imread('gauss_circle.png', 0) / 255.0
        circle_location_init = [5 + rng.randint(89), 5 + rng.randint(89)]
        sign_x = 2 * rng.randint(2) - 1
        sign_y = 2 * rng.randint(2) - 1
        x_velocity = sign_x * rng.randint(11)
        circle_velocity = [
            x_velocity / 2,
            sign_y * int(np.sqrt(max_velocity ** 2 - x_velocity ** 2)) / 2]

        circles.append(circle)
        circles_locations_init.append(circle_location_init)
        circles_velocities.append(circle_velocity)

    frames = np.zeros((time_steps, 100, 100))
    digit_location = digit_location_init
    clutters_locations = clutters_locations_init
    circles_locations = circles_locations_init
    for i in range(time_steps):
        digit_all_locations.append(digit_location)
        frames[i][digit_location[0] - 14:digit_location[0] + 14,
                  digit_location[1] - 14:digit_location[1] + 14] = digit
        digit_location = [digit_location[0] + digit_velocity[0],
                          digit_location[1] + digit_velocity[1]]

        if digit_location[0] > 85:
            digit_velocity[0] = -digit_velocity[0]
            digit_location[0] = 85

        if digit_location[1] > 85:
            digit_velocity[1] = -digit_velocity[1]
            digit_location[1] = 85

        if digit_location[0] < 14:
            digit_velocity[0] = -digit_velocity[0]
            digit_location[0] = 14

        if digit_location[1] < 14:
            digit_velocity[1] = -digit_velocity[1]
            digit_location[1] = 14

        for clutter, clutter_location, clutter_velocity, j in zip(
                clutters, clutters_locations,
                clutters_velocities, range(len(clutters))):
            frames[i][clutter_location[0] - 3:clutter_location[0] + 4,
                      clutter_location[1] - 3:clutter_location[1] + 4] = clutter
            clutter_location = [clutter_location[0] + clutter_velocity[0],
                                clutter_location[1] + clutter_velocity[1]]

            if clutter_location[0] > 95:
                clutter_velocity[0] = -clutter_velocity[0]
                clutter_location[0] = 95

            if clutter_location[1] > 95:
                clutter_velocity[1] = -clutter_velocity[1]
                clutter_location[1] = 95

            if clutter_location[0] < 3:
                clutter_velocity[0] = -clutter_velocity[0]
                clutter_location[0] = 3

            if clutter_location[1] < 3:
                clutter_velocity[1] = -clutter_velocity[1]
                clutter_location[1] = 3

            clutters_locations[j] = clutter_location
            clutters_velocities[j] = clutter_velocity

        for circle, circle_location, circle_velocity, j in zip(
                circles, circles_locations,
                circles_velocities, range(len(circles))):
            ones = np.ones(frames[i].shape)
            ones[circle_location[0] - 5:circle_location[0] + 6,
                 circle_location[1] - 5:circle_location[1] + 6] = circle
            frames[i] *= ones
            circle_location = [circle_location[0] + circle_velocity[0],
                               circle_location[1] + circle_velocity[1]]

            if circle_location[0] > 93:
                circle_velocity[0] = -circle_velocity[0]
                circle_location[0] = 93

            if circle_location[1] > 93:
                circle_velocity[1] = -circle_velocity[1]
                circle_location[1] = 93

            if circle_location[0] < 5:
                circle_velocity[0] = -circle_velocity[0]
                circle_location[0] = 5

            if circle_location[1] < 5:
                circle_velocity[1] = -circle_velocity[1]
                circle_location[1] = 5

            circles_locations[j] = circle_location
            circles_velocities[j] = circle_velocity

    all_frames.append((frames * 255).astype(np.uint8))
    all_locations.append(digit_all_locations)
print "Execution time: %f" % (time.time() - t0)

import h5py
f = h5py.File('/Tmp/pezeshki/dataset.hdf5', mode='w')
features = f.create_dataset(
    'features', (70000, time_steps, 100, 100), dtype='uint8')
locs = f.create_dataset(
    'locs', (70000, time_steps, 2), dtype='uint8')
targets = f.create_dataset(
    'targets', (70000,), dtype='uint8')

features[:, :, :, :] = np.vstack(
    [exp[np.newaxis] for exp in all_frames]).astype(np.uint8)
locs[:, :, :] = (np.vstack(
    [np.vstack(exp)[np.newaxis] for exp in all_locations])).astype(np.uint8)
targets[:] = np_targets.astype(np.uint8)

print 'hi'

features.dims[0].label = 'B'
features.dims[1].label = 'T'
features.dims[2].label = 'X'
features.dims[3].label = 'Y'
targets.dims[0].label = 'B'

split_array = np.empty(
    9,
    dtype=np.dtype([
        ('split', 'a', 5),
        ('source', 'a', 8),
        ('start', np.int64, 1),
        ('stop', np.int64, 1),
        ('indices', h5py.special_dtype(ref=h5py.Reference)),
        ('available', np.bool, 1),
        ('comment', 'a', 1)]))

split_array[0:3]['split'] = 'train'.encode('utf8')
split_array[3:6]['split'] = 'valid'.encode('utf8')
split_array[6:9]['split'] = 'test'.encode('utf8')

split_array[0:9:3]['source'] = 'features'.encode('utf8')
split_array[1:9:3]['source'] = 'locs'.encode('utf8')
split_array[2:9:3]['source'] = 'targets'.encode('utf8')

split_array[0:3]['start'] = 0
split_array[0:3]['stop'] = 50000
split_array[3:6]['start'] = 50000
split_array[3:6]['stop'] = 60000
split_array[6:9]['start'] = 60000
split_array[6:9]['stop'] = 70000

split_array[:]['indices'] = h5py.Reference()
split_array[:]['available'] = True
split_array[:]['comment'] = '.'.encode('utf8')
f.attrs['split'] = split_array

f.flush()
f.close()

from fuel.datasets import H5PYDataset
train_set = H5PYDataset('/Tmp/pezeshki/dataset.hdf5', which_sets=('train',))
