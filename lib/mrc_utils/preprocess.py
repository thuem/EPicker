import os
import cv2
import numpy as np

import mrc_utils.box as box
import mrc_utils.coord as coord
import mrc_utils.mrc as mrc
import mrc_utils.star as star
import mrc_utils.thi as thi
import mrc_utils.gen_box as gen_box

# from absl import app
# from absl import flags
reader_factory = {
    'star': star.read_star,
    'thi': thi.read_thi,
    'box': box.read_box,
    'coord': coord.read_coord
}


def downsample(inputs, use_factor=False, para1=None, para2=None):
    # This method executes a downsampling on the mrc
    # Inputs is a list of MrcData
    # If use_factor is True, para1 represents factor and para2 represents shape
    # Else para1 and para2 represents the target size(y, x)

    for i in range(len(inputs)):
        if use_factor:
            print("Processing %s ..." % (inputs[i].name))
            inputs[i].data = mrc.downsample_with_factor(
                inputs[i].data,
                factor=para1,
                shape=para2
            )
        else:
            print("Prcocessing %s ..." % (inputs[i].name))
            inputs[i].data = mrc.downsample_with_size(
                inputs[i].data,
                size1=para1,
                size2=para2
            )
    return inputs


def label_downsample(data, label, t, w, h):
    downsampled_label = []
    if t == 'star':
        for i in range(len(label)):
            w = h * data[i].data.shape[0] / data[i].data.shape[1]
            name = label[i].name
            content = star.downsample_with_size(
                label[i].content,
                (w / data[i].header[1], h / data[i].header[0])
            )
            downsampled_label.append(star.StarData(name, content))
    elif t == 'coord':
        for i in range(len(label)):
            w = h * data[i].data.shape[0] / data[i].data.shape[1]
            name = label[i].name
            content = coord.downsample_with_size(
                label[i].content,
                (w / data[i].header[1], h / data[i].header[0])
            )
            downsampled_label.append(coord.CoordData(name, content))
    elif t == 'box':
        for i in range(len(label)):
            w = h * data[i].data.shape[0] / data[i].data.shape[1]
            name = label[i].name
            content = box.downsample_with_size(
                label[i].content,
                (w / data[i].header[1], h / data[i].header[0])
            )
            downsampled_label.append(box.BoxData(name, content))
    elif t == 'thi':
        for i in range(len(label)):
            w = h * data[i].data.shape[0] / data[i].data.shape[1]
            name = label[i].name
            content = thi.downsample(
                label[i].content,
                (w / data[i].header[1], h / data[i].header[0])
            )
            downsampled_label.append(thi.ThiData(name, content))
    else:
        print('A valid type of label is required: star | coord | box | thi')
    return downsampled_label


def write_label(label):
    # TODO: merge all types of label
    pass


def read_label(label_path, label_type):
    if label_type == 'star':
        return star.read_all_star(label_path)
    elif label_type == 'coord':
        return coord.read_all_coord(label_path)
    elif label_type == 'box':
        print('reading box')
        return box.read_all_box(label_path)
    elif label_type == 'thi':
        print('reading thi')
        return thi.read_all_thi(label_path)
    else:
        print('A valid type is required: star | coord | box')
        return []


def load_and_downsample(path, target_size):
    with open(path, "rb") as f:
        content = f.read()
    data, header, _ = mrc.parse(content=content)
    name = '.'.join(path.split('/')[-1].split('.')[:-1])
    # averge frame
    if header[2] > 1:
        avg_mrc = np.zeros_like(data[0, ...], dtype='float')
        for j in range(header[2]):
            avg_mrc += data[j, ...]
        avg_mrc /= header[2]
        data = avg_mrc
    data = mrc.downsample_with_size(data, int(data.shape[0] / data.shape[1] * 1024), target_size)
    return mrc.MrcData(name, data, header)


def process(opt):
    path = opt.data
    isdir = os.path.isdir(opt.data)

    if opt.data_type == 'mrc':
        mrc_files = []
        mrc_data = []
        if isdir:
            for file in os.listdir(path):
                if file.endswith('.mrc'):
                    # print("Loading %s ..." % (file))
                    # data = load_and_downsample(os.path.join(path, file), opt.target_size)
                    mrc_files.append(file)
        elif opt.data.endswith('.thi'):
            with open(opt.data) as f:
                while (True):
                    line = f.readline()
                    if not line:
                        break
                    if line.startswith('['):
                        continue
                    file = line.rstrip('\n')
                    file = file.rstrip(' ')
                    file = file.lstrip(' ')
                    print('Loading %s ...' % (file))
                    # data = load_and_downsample(file, opt.target_size)
                    # mrc_data.append(data)
                    mrc_files.append(file)
        else:
            print('Invalid data: opt.data should be a directory or thi file')
            return
        # mrc_data.sort(key=lambda m: m.name)
        # load label
        label = []
        if isdir:
            label_path = opt.label if opt.label else opt.data
            import re
            label_files = [re.sub('\.' + opt.label_type + '$', '', l) for l in os.listdir(label_path) if
                           l.endswith(opt.label_type)]
            mrc_files = [re.sub('\.mrc$', '', m) for m in mrc_files]
            intersection = list(set(mrc_files).intersection(set(label_files)))
            union = list(set(mrc_files) | set(label_files))
            difference = list(set(union).difference(intersection))
            # mrc_files = [ m+'.mrc' for m in intersection ]
            # label_files = [ l+'.'+opt.label_type for l in intersection ]
            # print('data files:', mrc_files)
            # print('label_files:', label_files)
            # print('intersection:', intersection)
            for d in difference:
                if d in mrc_files:
                    print('Warning: No matching label file for %s.mrc' % (d))
                elif d in label_files:
                    print('Warning: No matching data file for %s.%s' % (d, opt.label_type))

            mrc_files = [m + '.mrc' for m in intersection]
            label_files = [l + '.' + opt.label_type for l in intersection]

            reader = reader_factory[opt.label_type]

            if opt.mode == 'fiber':
                fiber_label_path = os.path.join(label_path, 'processed_labels')
                if not os.path.exists(fiber_label_path):
                    os.makedirs(fiber_label_path)
                for l in label_files:
                    if l.endswith('thi'):
                        with open(os.path.join(label_path, l)) as f:
                            header = f.readlines()
                        if not header:
                            continue
                        elif 'Group' not in header[0].split():
                            os.system('cp %s %s' % (os.path.join(label_path, l), os.path.join(fiber_label_path, l)))
                        else:
                            group = gen_box.read_thi(os.path.join(label_path, l))
                            boxes = gen_box.place_boxes(group)
                            gen_box.write_thi(boxes, os.path.join(fiber_label_path, l))
                label_path = fiber_label_path

            for l in label_files:
                print('Loading %s ...' % (l))
                label.append(reader(os.path.join(label_path, l)))

            for m in mrc_files:
                print('Loading %s ...' % (m))
                mrc_data.append(load_and_downsample(os.path.join(opt.data, m), opt.target_size))

        # label = read_label(label_path, opt.label_type)
        elif opt.data.endswith('.thi'):
            label_path = opt.label if opt.label else opt.data
            label_files = [l.rstrip('.' + opt.label_type) for l in os.listdir(label_path) if l.endswith(opt.label_type)]
            mrc_names = [m.split('/')[-1].rstrip('.mrc') for m in mrc_files]
            name2file = {}
            for i in range(len(mrc_names)):
                name2file[mrc_names[i]] = mrc_files[i]
            intersection = list(set(mrc_names).intersection(set(label_files)))
            union = list(set(mrc_names) | set(label_files))
            difference = list(set(union).difference(intersection))

            for d in difference:
                if d in mrc_names:
                    print('Warning: No matching label file for %s' % (name2file[d]))
                elif d in label_files:
                    print('Warning: No matching data file for %s.%s' % (d, opt.label_type))

            mrc_files = [name2file[m] for m in intersection]
            label_files = [l + '.' + opt.label_type for l in intersection]

            reader = reader_factory[opt.label_type]
            if opt.mode == 'fiber':
                fiber_label_path = os.path.join(label_path, 'processed_labels')
                if not os.path.exists(fiber_label_path):
                    os.makedirs(fiber_label_path)
                for l in label_files:
                    if l.endswith('.thi'):
                        header = None
                        with open(l) as f:
                            header = f.readline()
                        if not header:
                            continue
                        elif len(header.split()) < 6:
                            os.system('cp %s %s' % (l, os.path.join(fiber_label_path, l)))
                        else:
                            group = gen_box.read_thi(os.path.join(label_path, l))
                            boxes = gen_box.place_boxes(group)
                            gen_box.write_thi(boxes, os.path.join(fiber_label_path, l))
                label_path = fiber_label_path

            for l in label_files:
                print('Loading %s ...' % (l))
                label.append(reader(os.path.join(label_path, l)))

            for m in mrc_files:
                print('Loading %s ...' % (m))
                mrc_data.append(load_and_downsample(os.path.join(opt.data, m), opt.target_size))
            # label = read_label(label_path, opt.label_type)
            # mrc_name = []
            # for m in mrc_data:
            #    mrc_name.append(m.name)
            # intersection = []
            # for l in label:
            #    if l.name in mrc_name:
            #        intersection.append(l)
            # label = intersection
        label.sort(key=lambda l: l.name)
        mrc_data.sort(key=lambda l: l.name)
        # debug:
        # for k in range(len(mrc_data)):
        #    print(mrc_data[k].name, '\t', label[k].name)

        downsampled_label = label_downsample(
            mrc_data, label,
            opt.label_type,
            opt.target_size * mrc_data[0].data.shape[0] / mrc_data[0].data.shape[1],
            opt.target_size
        )
        image_path = os.path.join('./', opt.exp_id, 'images')
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        for m in mrc_data:
            mrc.save_image(m.data, os.path.join(image_path, m.name), opt.target_size, f='png', verbose=True)
        print(len(mrc_data), 'micrographs loaded')
        num_train = int(opt.train_pct / 100.0 * len(mrc_data))
        num_val = int(opt.val_pct / 100.0 * len(mrc_data))
        num_test = int(opt.test_pct / 100.0 * len(mrc_data))
        train = downsampled_label[0:num_train]
        val = downsampled_label[num_train:num_train + num_val]
        test = downsampled_label[num_train + num_val:num_train + num_val + num_test]
        print('Creating COCO annotations')
        # print(train, val, test)
        anno_path = os.path.join('./', opt.exp_id, 'annotations')
        if not os.path.exists(anno_path):
            os.makedirs(anno_path)

        star.star2coco(train, image_path, 4 * opt.particle_size, os.path.join(anno_path, 'train'))
        star.star2coco(val, image_path, 4 * opt.particle_size, os.path.join(anno_path, 'val'))
        star.star2coco(test, image_path, 4 * opt.particle_size, os.path.join(anno_path, 'test'))
    elif opt.data_type == 'png':
        image_path = os.path.join(opt.data, 'images')
        anno_path = os.path.join(opt.data, 'annotations')
    # print('Calculating mean and var of this dataset')
    # return get_mean_and_var(image_path, 1024*1024)


def get_mean_and_var(filepath, pixels):
    dir = os.listdir(filepath)
    # print(dir)
    r, g, b = 0, 0, 0
    for idx in range(len(dir)):
        filename = dir[idx]
        img = cv2.imread(os.path.join(filepath, filename)) / 255.0
        r = r + np.sum(img[:, :, 0])
        g = g + np.sum(img[:, :, 1])
        b = b + np.sum(img[:, :, 2])

    pixels = len(dir) * pixels
    r_mean = r / pixels
    g_mean = g / pixels
    b_mean = b / pixels

    r, g, b = 0, 0, 0
    for i in range(len(dir)):
        filename = dir[i]
        img = cv2.imread(os.path.join(filepath, filename)) / 255.0
        r = r + np.sum((img[:, :, 0] - r_mean) ** 2)
        g = g + np.sum((img[:, :, 1] - g_mean) ** 2)
        b = b + np.sum((img[:, :, 2] - b_mean) ** 2)

    r_var = np.sqrt(r / pixels)
    g_var = np.sqrt(g / pixels)
    b_var = np.sqrt(b / pixels)
    r_mean = np.float32(r_mean)
    g_mean = np.float32(g_mean)
    b_mean = np.float32(b_mean)
    r_var = np.float32(r_var)
    g_var = np.float32(g_var)
    b_bar = np.float32(b_var)
    print("r_mean is %f, g_mean is %f, b_mean is %f" % (r_mean, g_mean, b_mean))
    print("r_var is %f, g_var is %f, b_var is %f" % (r_var, g_var, b_var))
    return [r_mean, g_mean, g_mean], [r_var, g_var, b_var]


'''
def main(argv):
    del argv
    data_path = FLAGS.data_path
    label_path = FLAGS.label_path
    data_dst = FLAGS.data_dst_path
    label_dst = FLAGS.label_dst_path
    target_size = FLAGS.target_size
    label_type = FLAGS.label_type

    mrc_data = mrc.load_all_mrc(data_path)
    #averge frame
    for i in range(len(mrc_data)):
        if mrc_data[i].header[2] < 2:
            break
        avg_mrc = np.zeros_like(mrc_data[i].data[0,...])
        for j in range(mrc_data[i].header[2]):
            avg_mrc += mrc_data[i].data[j, ...]
        avg_mrc /= mrc_data[i].header[2]
        mrc_data[i].data = avg_mrc
    label = read_label(label_path, label_type)
    #debug:
    for k in range(len(mrc_data)):
        print(mrc_data[k].name, '\t', label[k].name)
    mrc_data = downsample(mrc_data, False, para1=target_size, para2=target_size)
    downsampled_label = label_downsample(
        mrc_data, label, 
        label_type, 
        target_size, target_size
    )
    print(len(downsampled_label))

    mrc.write_mrc(mrc_data, dst=data_dst)
    #write_label(downsampled_label, label_type)
    print('writing label...')
    star.write_star(downsampled_label, dst=label_dst)  

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string("data_path", None, "path of data(mrc, etc.)")
    flags.DEFINE_string("label_path", None, "path of labels(star, etc.)")
    flags.DEFINE_string("data_dst_path", None, "target to store processed data")
    flags.DEFINE_string("label_dst_path", None, "target to store processed labels")
    flags.DEFINE_string("label_type", "star", "type of label, star | coord | box")
    flags.DEFINE_integer("target_size", 1024, "the size of processed data")
    flags.mark_flag_as_required("data_path")
    flags.mark_flag_as_required("label_path")
    flags.mark_flag_as_required("data_dst_path")
    flags.mark_flag_as_required("label_dst_path")
    app.run(main)
'''
