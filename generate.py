from collections import OrderedDict
import h5py
import subprocess
import itertools
from PIL import Image
import cPickle
import progressbar
import numpy as np
import scipy.misc
from StringIO import StringIO
import fuel.datasets

seed = 1234
rng = np.random.RandomState(seed)

tmpfile = "mnist_temp.npz"
try:
    data = np.load(tmpfile)
    features, targets = data["features"], data["targets"]
    data.close()
except IOError:
    print "preprocessing"
    with open("/data/lisa/data/mnist/mnist.pkl", "rb") as f:
        dataset = cPickle.load(f)
        features, targets = [
            np.concatenate(xs, axis=0) for xs in zip(*dataset)]
        dataset = None
        features = features.reshape((features.shape[0], 28, 28))
        np.savez(tmpfile, features=features, targets=targets)

video_shape = np.array([20, 100, 100])
frame_shape = video_shape[1:]
n_distractors = 8
distractor_shape = np.array([8, 8])
occlusion_radius = 6
occlusion_step = 12
# uniform speed distribution in pixels per frame
speed_min = 0.5
speed_max = 3.0
subpixel = True


def get_distractors(n):
    i = rng.randint(features.shape[0], size=(n,))
    # don't take too close to edge to avoid empty patches
    margin = 2
    maxtopleft = (np.array(features.shape[1:]) -
                  2 * margin - distractor_shape)
    # randint doesn't support high=some_shape, so use random_sample
    topleft = np.round(rng.random_sample((n, 2)) * maxtopleft).astype(int)
    y = (margin + topleft[:, 0, np.newaxis] +
         np.arange(distractor_shape[0])[np.newaxis, ...])
    x = (margin + topleft[:, 1, np.newaxis] +
         np.arange(distractor_shape[1])[np.newaxis, ...])
    return features[i[:, np.newaxis, np.newaxis],
                    y[:, :, np.newaxis],
                    x[:, np.newaxis, :]]


def place(frame, x, patch):
    x = np.round(x)
    pa = np.clip(- x, (0, 0), patch.shape)
    pb = np.clip(frame_shape - x, (0, 0), patch.shape)
    fa = np.clip(x, (0, 0), frame_shape)
    fb = np.clip(x + patch.shape, (0, 0), frame_shape)
    frame[fa[0]:fb[0], fa[1]:fb[1]] += patch[pa[0]:pb[0], pa[1]:pb[1]]


def random_trajectories(patches):
    n = patches.shape[0]
    patch_shape = patches.shape[1:]
    t = rng.randint(video_shape[0], size=(n,))
    xt = np.round(rng.random_sample((n, 2)) * (
        frame_shape - patch_shape)).astype(np.float32)
    speed = rng.uniform(speed_min, speed_max, size=(n, 1)).astype(np.float32)
    direction = rng.uniform(-1., 1., size=(n, 2)).astype(np.float32)
    v = speed * direction / np.linalg.norm(direction)
    return t, xt, v, patches


class Gahh(Exception):
    pass


def _roll_into_dimslice(d, n):
    if abs(d) >= n:
        # believe it or not, checking in here and raising a
        # control-flow exception is much faster than checking before
        # the loop in roll_into
        raise Gahh()
    leftright = slice(0, n - abs(d)), slice(abs(d), n)
    if d >= 0:
        return leftright[1], leftright[0]
    else:
        return leftright


def roll_into(dest, src, ds):
    # assert dest.shape == src.shape
    # assert np.allclose(0, dest)
    try:
        dest_indices, src_indices = zip(*(
            _roll_into_dimslice(int(d), dim)
            for d, dim in zip(ds, dest.shape)))
        dest[dest_indices] = src[src_indices]
    except Gahh:
        pass

if subpixel:
    import scipy.ndimage.interpolation

    def shift(dest, src, dx):
        scipy.ndimage.interpolation.shift(src, dx, output=dest)
else:

    def shift(dest, src, dx):
        roll_into(dest, src, dx)


def render_trajectories(video, ts, xts, vs, patches):
    scratch = video * 0  # allocate once
    for i, (t, xt, v, patch) in enumerate(zip(ts, xts, vs, patches)):
        scratch.fill(0)
        place(scratch[t], xt, patch)
        for k in xrange(video_shape[0]):
            if k == t:
                continue
            dx = (k - t) * v
            shift(scratch[k], scratch[t], dx)
        video += scratch
    return video

if subpixel:
    # vertical occlusions; generate once then rotate as desired
    occlusions = np.tile(
        np.concatenate(
            [np.zeros([occlusion_radius], dtype=np.float32),
             np.ones([occlusion_step - occlusion_radius], dtype=np.float32)]),
        # at least (1.5 + sqrt(2)) times as large:
        # * 1.5 because because scipy...affine_transform takes
        #   `offset` to be both the pivot of the rotation and the
        #   topleft corner of the output
        # * sqrt(2) because we want the vertical to be able to span
        #   the diagonal
        (3 * frame_shape[0], 3 * (frame_shape[1] / occlusion_step)))
    # storage for rotated occlusions
    occlusions_storage = np.zeros(frame_shape, dtype=np.float32)
    occlusions_pivot = np.array(occlusions.shape) / 2

    def render_occlusions(video, *digit_trajectory):
        [(t, xt, v, patch)] = zip(*digit_trajectory)
        v /= np.linalg.norm(v)
        scipy.ndimage.interpolation.affine_transform(
            occlusions,
            # rotate to be perpendicular to the velocity of the digit
            np.array([[v[1], -v[0]],
                      [v[0], v[1]]]),
            offset=occlusions_pivot,
            output_shape=tuple(frame_shape),
            output=occlusions_storage)
        # occlusions are stationary
        video *= occlusions_storage
        return video
else:
    def render_occlusions(video, *digit_trajectory):
        [(t, xt, v, patch)] = zip(*digit_trajectory)
        # use axis-aligned bars for occlusion, perpendicular to the
        # velocity of the digit
        dim = v.argmax()
        index = list(map(slice, frame_shape))
        offset = rng.randint(occlusion_step)
        for j in xrange(occlusion_radius):
            index[dim] = slice(offset + j, None, occlusion_step)
            video[np.index_exp[:] + tuple(index)] = 0
        return video


def write_video(video, rate, filename):
    cmdstring = ('ffmpeg',
                 '-y',
                 '-r', '%d' % rate,
                 '-f', 'image2pipe',
                 '-vcodec', 'mjpeg',
                 '-i', 'pipe:',
                 '-vcodec', 'libxvid',
                 filename)
    p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE, shell=False)
    for frame in video:
        im = Image.fromarray(np.uint8(frame * 255))
        p.stdin.write(im.tostring('jpeg', 'L'))
    p.stdin.close()


def plot_frames(video):
    import matplotlib.pyplot as plt
    for frame in video:
        plt.imshow(
            frame, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        plt.show()


def generate_example(digit):
    digit_trajectory = random_trajectories(digit[np.newaxis, ...])
    video = np.zeros(video_shape, dtype=np.float32)
    video = render_trajectories(video, *digit_trajectory)
    video = render_occlusions(video, *digit_trajectory)
    video = render_trajectories(
        video, *random_trajectories(get_distractors(n_distractors)))
    video = np.clip(video, 0.0, 1.0)
    return video


def main():
    filename = "cmv%ix%ix%i_jpeg_%i.hdf5" % (tuple(video_shape) + (seed,))

    partition = OrderedDict([
        ("train", 50000),
        ("valid", 10000),
        ("test", 10000)])
    size = sum(partition.values())

    f = h5py.File(filename, mode='w')

    # we represent videos by ranges of frames
    f.create_dataset("videos", (size, 2), dtype=np.uint64)
    f.create_dataset("targets", (size,), dtype=np.uint8)

    f.create_group("frames")

    vi = 0
    for which_set, set_size in partition.items():
        widgets = [progressbar.Counter(), " ",
                   "(", progressbar.Percentage(), ")",
                   progressbar.Bar(), " ",
                   progressbar.Timer(), " ",
                   progressbar.ETA()]
        progress = progressbar.ProgressBar(widgets=widgets,
                                           maxval=set_size)
        progress.start()

        f["frames"].create_dataset(
            which_set, (set_size * video_shape[0],),
            dtype=h5py.special_dtype(vlen=np.uint8))
        fb = 0
        for i, (digit, target) in zip(xrange(set_size),
                                      itertools.cycle(zip(features, targets))):
            frames = generate_example(digit)
            fa, fb = fb, fb + len(frames)
            f["frames"][which_set][fa:fb] = map(asjpeg, frames)
            f["videos"][vi] = [fa, fb]
            f["targets"][vi] = target
            vi += 1

            # optionally write out some example videos
            if False:
                if i == 10:
                    return
                framerate = 5
                write_video(frames, framerate, "%i.avi" % i)

            progress.update(i)
        progress.finish()

    ranges = np.cumsum([0] + list(partition.values()))
    split_dict = OrderedDict(
        (which_set, OrderedDict([
            ("videos", (sa, sb)),
            ("targets", (sa, sb)),
        ]))
        for which_set, sa, sb in zip(
            partition.keys(), ranges[:-1], ranges[1:]))
    print split_dict
    f.attrs["split"] = fuel.datasets.H5PYDataset.create_split_array(split_dict)

    f.flush()
    f.close()


def asjpeg(image):
    # we store the frames as jpeg and do the decompression on the fly.
    data = StringIO()
    Image.fromarray(image).convert("L").save(data, format="JPEG")
    data.seek(0)
    return np.fromstring(data.read(), dtype="uint8")

main()
