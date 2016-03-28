import numpy as np
from PIL import Image
import h5py
from fuel.datasets import H5PYDataset

# size: 14105
with open('dish.csv', 'rb') as csvfile:
    line = csvfile.readline()
    array = line.split(',')
    all_labels = [int(element) for element in array]

# size: 14105
with open('starts.csv', 'rb') as csvfile:
    line = csvfile.readline()
    array = line.split(',')
    all_starts = [int(element) for element in array]

# size: 14105
with open('ends.csv', 'rb') as csvfile:
    line = csvfile.readline()
    array = line.split(',')
    all_ends = [int(element) for element in array]

# size: 14105
with open('names.csv', 'rb') as csvfile:
    all_folders = []
    for line in csvfile.readlines():
        all_folders.append(line[:-1])

# size: 14105
with open('ingredients.txt', 'rb') as csvfile:
    ingredients = []
    for line in csvfile.readlines():
        ingredients.append(line[:-1])

# size:
with open('sequencesTest.txt', 'rb') as csvfile:
    test_folders = []
    for line in csvfile.readlines():
        test_folders.append(line[:-1])

# size:
with open('sequencesVal.txt', 'rb') as csvfile:
    valid_folders = []
    for line in csvfile.readlines():
        valid_folders.append(line[:-1])

test_labels = []
for folder in test_folders:
    idx = all_folders.index(folder)
    test_labels.append(all_labels[idx])
test_labels = list(set(test_labels))
assert len(test_labels) == 31

# clean means that a video label is also appread in the test set
all_clean_folders = []
all_clean_labels = []
all_clean_starts = []
all_clean_ends = []
for folder, label, start, end in zip(
        all_folders, all_labels, all_starts, all_ends):
    if label in test_labels:
        all_clean_folders.append(folder)
        all_clean_starts.append(start)
        all_clean_ends.append(end)
        all_clean_labels.append(label)

unites = []
mapping = {}
i = 0
for folder in list(set(all_clean_folders)):
    mapping[folder] = i
    i = i + 1
for folder in all_clean_folders:
    unites.append(mapping[folder])

if True:
    all_clips = []
    general_path = '/data/lisatmp4/negar/datasets/originalsizeimages/'
    alls = len(all_clean_folders)
    print alls
    for i, folder, start, end in zip(
            range(alls), all_clean_folders, all_clean_starts, all_clean_ends):
        if i % 20 == 0:
            print i
        if folder == 's37-d7':
            folder = 's37-d74'
        path = general_path + folder + '-cam-002/'
        frame_indices = np.linspace(start - 1, end - 1, 12, dtype='int')

        step = frame_indices[1] - frame_indices[0]

        jpeg_files = ['image' + str(n) + '.jpg' for n in frame_indices]
        a_clip = []

        for jpeg_file in jpeg_files:
            img = np.array(Image.open(path + jpeg_file)).astype('uint8')
            # size: 1 x 125 x 200 x 3
            a_clip.append(img[np.newaxis])
        all_clips.append(np.vstack(a_clip).astype('uint8'))
else:
    all_clips = np.load('all_clips.npz')['arr_0']

train_clips = []
valid_clips = []
test_clips = []
train_labels = []
valid_labels = []
test_labels = []
train_unites = []
valid_unites = []
test_unites = []

for folder, label, clip, unite in zip(
        all_clean_folders, all_clean_labels, all_clips, unites):
    if folder in test_folders:
        test_clips.append(clip)
        test_labels.append(label)
        test_unites.append(unite)
    elif folder in valid_folders:
        valid_clips.append(clip)
        valid_labels.append(label)
        valid_unites.append(unite)
    else:
        train_clips.append(clip)
        train_labels.append(label)
        train_unites.append(unite)

all_clips_sorted = train_clips + valid_clips + test_clips
all_labels_sorted = train_labels + valid_labels + test_labels
all_unites_sorted = train_unites + valid_unites + test_unites

f = h5py.File('/Tmp/pezeshki/cooking_orig.hdf5', mode='w')
features = f.create_dataset(
    'features', (3 * 8585, 12, 1224, 1624, 3), dtype='uint8')
targets = f.create_dataset(
    'targets', (3 * 8585,), dtype='uint8')
unites = f.create_dataset(
    'unites', (3 * 8585,), dtype='uint8')

features[:, :, :, :] = np.vstack(
    [exp[np.newaxis] for exp in all_clips_sorted]).astype(np.uint8)
targets[:] = np.array(all_labels_sorted).astype(np.uint8)
unites[:] = np.array(all_unites_sorted).astype(np.uint8)

print 'hi'

features.dims[0].label = 'B'
features.dims[1].label = 'T'
features.dims[2].label = 'X'
features.dims[3].label = 'Y'
features.dims[4].label = 'C'
targets.dims[0].label = 'B'
unites.dims[0].label = 'B'

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
split_array[1:9:3]['source'] = 'targets'.encode('utf8')
split_array[2:9:3]['source'] = 'unites'.encode('utf8')

split_array[0:3]['start'] = 0
split_array[0:3]['stop'] = 3 * 6038
split_array[3:6]['start'] = 3 * 6038
split_array[3:6]['stop'] = 3 * 6483
split_array[6:9]['start'] = 3 * 6483
split_array[6:9]['stop'] = 3 * 8585

split_array[:]['indices'] = h5py.Reference()
split_array[:]['available'] = True
split_array[:]['comment'] = '.'.encode('utf8')
f.attrs['split'] = split_array

f.flush()
f.close()

train_set = H5PYDataset('/Tmp/pezeshki/cookingorig.hdf5', which_sets=('train',))

import ipdb; ipdb.set_trace()
