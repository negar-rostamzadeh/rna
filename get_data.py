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
        ingredients.append(line[1:-2])

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
all_clean_ingredients = []
for folder, label, start, end, ingredient in zip(
        all_folders, all_labels, all_starts, all_ends, ingredients):
    if label in test_labels:
        all_clean_folders.append(folder)
        all_clean_starts.append(start)
        all_clean_ends.append(end)
        all_clean_labels.append(label)
        all_clean_ingredients.append(ingredient)

# clean2 means that the ingredient is consistent with the label
all_clean2_folders = []
all_clean2_labels = []
all_clean2_starts = []
all_clean2_ends = []
all_clean2_ingredients = []

mapping = {21: 0, 23: 1, 25: 2, 26: 3, 27: 4, 28: 5, 29: 6, 31: 7,
           34: 8, 35: 9, 39: 10, 40: 11, 41: 12, 42: 13, 43: 14,
           45: 15, 46: 16, 48: 17, 49: 18, 50: 19, 51: 20, 52: 21,
           53: 22, 54: 23, 55: 24, 63: 25, 69: 26, 70: 27, 71: 28,
           73: 29, 74: 30}

classes = {0: 'cucumber',
           1: 'carrots',
           2: 'bread',
           3: 'cauliflower',
           4: 'onion',
           5: 'orange',
           6: 'herbs',
           7: 'garlic',
           8: 'ginger',
           9: 'plum',
           10: 'leeks',
           11: 'lime',
           12: 'pomegranate',
           13: 'broccli',
           14: 'potato',
           15: 'pepper',
           16: 'pineapple',
           17: 'chilli',
           18: 'pasta',
           19: 'scrambled egg',
           20: 'broad beans',
           21: 'kiwi',
           22: 'avocado',
           23: 'mango',
           24: 'figs',
           25: 'toaster',
           26: 'separating egg',
           27: 'juicing orange',
           28: 'hot dog',
           29: 'tea',
           30: 'coffee'}

relevant_ingredients = {'cucumber': ['cucumber', 'cucumbercutting-boardplate'],
                        'carrots': ['carrot', 'front-peeler'],
                        'bread': ['plastic-paper-bag', 'bread', 'bread-knife'],
                        'cauliflower': ['cauliflower', 'cauliflowercauliflower'],
                        'onion': ['onion', 'onionpeel', 'peel', 'bottleonion', 'onionnet-bag', 'onionwater'],
                        'orange': ['orange', 'orangepeel', 'orangeorange'],
                        'herbs': ['parsleybundle', 'parsley', 'bundle', 'flower-pot', 'chive', 'oregano', 'chiveoreganoparsley', 'parsleyparsley'],
                        'garlic': ['garlic-press', 'garlic-bulb', 'garlic-clove', 'garlic', 'garlic-cloveplate'],
                        'ginger': ['ginger', 'gingerspice'],
                        'plum': ['plum', 'plumplum'],
                        'leeks': ['leek', 'leekleek'],
                        'lime': ['squeezer', 'lime', 'squeezerspoon'],
                        'pomegranate': ['pomegranate', 'arilspomegranate', 'arils', 'pomegranatearils'],
                        'broccli': ['broccoli', 'broccolibroccoli'],
                        'potato': ['potato'],
                        'pepper': ['pepper', 'seedpepper', 'peppercorn'],
                        'pineapple': ['pineapple', 'peel', 'peelpineapple', 'saltpineapple'],
                        'chilli': ['chilli', ],
                        'pasta': ['pasta', 'pastawater', 'stove', 'pot', 'colander', 'potpasta'],
                        'scrambled egg': ['egg', 'egg-white', 'yolk', 'eggshell', 'oil', 'frying-panspatula', 'butteregg'],
                        'broad beans': ['green-beans'],
                        'kiwi': ['kiwi', ],
                        'avocado': ['avocadostone', 'avocado'],
                        'mango': ['mango', 'front-peeler', ],
                        'figs': ['figfig', 'fig'],
                        'toaster': ['toaster', 'bread', ],
                        'separating egg': ['egg', 'egg-white', 'yolk', 'eggshell', 'oil', 'frying-panspatula', 'butteregg'],
                        'juicing orange': ['orange', 'orangepeel'],
                        'hot dog': ['ketchup', 'mustard', 'ketchupmustard', 'hot-dogwater', 'hot-dog'],
                        'tea': ['tea', 'tea-herbs', 'kettle-power-basewater-kettle', 'water-kettle', 'sugar', 'kettle-power-base', 'teasugar', 'teaspoon'],
                        'coffee': ['coffee', 'milk', 'coffee-machine', 'coffee-powder', 'coffee-containerpaper-box', 'coffee-filtercoffee-container', 'coffee-container', 'sugar', 'coffee-filter']}

for folder, label, start, end, ingredient in zip(
        all_clean_folders,
        all_clean_labels,
        all_clean_starts,
        all_clean_ends,
        all_clean_ingredients):
    if ingredient in relevant_ingredients[classes[mapping[label]]]:
        all_clean2_folders.append(folder)
        all_clean2_starts.append(start)
        all_clean2_ends.append(end)
        all_clean2_labels.append(label)
        all_clean2_ingredients.append(ingredient)

all_clean_folders = all_clean2_folders
all_clean_labels = all_clean2_labels
all_clean_starts = all_clean2_starts
all_clean_ends = all_clean2_ends

unites = []
mapping = {}
i = 0
for folder in list(set(all_clean_folders)):
    mapping[folder] = i
    i = i + 1
for folder in all_clean_folders:
    unites.append(mapping[folder])


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
    frame_indices = np.linspace(start - 1, end - 1, 10, dtype='int')

    step = frame_indices[1] - frame_indices[0]

    jpeg_files = ['image' + str(n) + '.jpg' for n in frame_indices]
    a_clip = []

    for jpeg_file in jpeg_files:
        img = np.array(Image.open(path + jpeg_file)).astype('uint8')
        # size: 1 x 125 x 200 x 3
        a_clip.append(img[np.newaxis])
    all_clips.append(np.vstack(a_clip).astype('uint8'))

train_clips = []
valid_clips = []
test_clips = []
train_labels = []
valid_labels = []
test_labels = []
train_unites = []
valid_unites = []
test_unites = []

for i, folder, label, clip, unite in zip(
        range(len(all_clean_folders)), all_clean_folders,
        all_clean_labels, all_clips, unites):
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
    all_clips.remove(clip)

train_clips = train_clips + valid_clips + test_clips
del valid_clips[:]
del valid_clips
del test_clips[:]
del test_clips
all_labels_sorted = train_labels + valid_labels + test_labels
all_unites_sorted = train_unites + valid_unites + test_unites

np.save('/data/lisatmp4/pezeshki/part1', np.vstack(
    [exp[np.newaxis].astype(np.uint8) for
     exp in train_clips[:1000]]).astype(np.uint8))
del train_clips[:1000]
np.save('/data/lisatmp4/pezeshki/part2', np.vstack(
    [exp[np.newaxis].astype(np.uint8) for
     exp in train_clips[1000:2000]]).astype(np.uint8))
del train_clips[1000]
np.save('/data/lisatmp4/pezeshki/part3', np.vstack(
    [exp[np.newaxis].astype(np.uint8) for
     exp in train_clips[2000:3000]]).astype(np.uint8))
del train_clips[:1000]
np.save('/data/lisatmp4/pezeshki/part4', np.vstack(
    [exp[np.newaxis].astype(np.uint8) for
     exp in train_clips[3000:]]).astype(np.uint8))

np.save('/data/lisatmp4/pezeshki/all_labels_sorted', all_labels_sorted)
np.save('/data/lisatmp4/pezeshki/all_unites_sorted', all_unites_sorted)
import ipdb; ipdb.set_trace()

f = h5py.File('/Tmp/pezeshki/cooking_orig.hdf5', mode='w')
features = f.create_dataset(
    'features', (3 * 4119, 10, 1224, 1624, 3), dtype='uint8')
targets = f.create_dataset(
    'targets', (3 * 4119,), dtype='uint8')
unites = f.create_dataset(
    'unites', (3 * 4119,), dtype='uint8')

import ipdb; ipdb.set_trace()

try:
    train_clips = np.vstack(
        [exp[np.newaxis].astype(np.uint8) for
         exp in train_clips]).astype(np.uint8)
    features[:, :, :, :] = train_clips
    targets[:] = np.array(all_labels_sorted).astype(np.uint8)
    unites[:] = np.array(all_unites_sorted).astype(np.uint8)

except Exception, e:
    import ipdb; ipdb.set_trace()

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
split_array[0:3]['stop'] = 3 * 2896
split_array[3:6]['start'] = 3 * 2896
split_array[3:6]['stop'] = 3 * 3110
split_array[6:9]['start'] = 3 * 3110
split_array[6:9]['stop'] = 3 * 4119

split_array[:]['indices'] = h5py.Reference()
split_array[:]['available'] = True
split_array[:]['comment'] = '.'.encode('utf8')
f.attrs['split'] = split_array

f.flush()
f.close()

train_set = H5PYDataset('/Tmp/pezeshki/cookingorig.hdf5', which_sets=('train',))

import ipdb; ipdb.set_trace()
