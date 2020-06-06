import tensorflow as tf
import yaml
import os
from object_detection.utils import dataset_util, label_map_util


flags = tf.app.flags
flags.DEFINE_string(
    'data_dir', '', 'Specify root directory to raw dataset and/or to a .yaml file. Seperate multiple datasets with a comma.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', '',
                    'Path to label map proto e.g.: data/label_map.pbtxt')
FLAGS = flags.FLAGS


def create_tf_example(example, label_map_dict):

    # Bosch
    height = 720  # Image height
    width = 1280  # Image width

    # Filename of the image. Empty if image is not from file
    filename = example['path']
    filename = filename.encode()

    with tf.gfile.GFile(example['path'], 'rb') as fid:
        encoded_image = fid.read()

    image_format = 'png'.encode()

    # List of normalized left x coordinates in bounding box (1 per box)
    xmins = []
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    # List of normalized top y coordinates in bounding box (1 per box)
    ymins = []
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    for box in example['boxes']:
        # if box['occluded'] is False:
        #print("adding box")
        xmins.append(float(box['x_min'] / width))
        xmaxs.append(float(box['x_max'] / width))
        ymins.append(float(box['y_min'] / height))
        ymaxs.append(float(box['y_max'] / height))
        classes_text.append(box['label'].encode())
        classes.append(int(label_map_dict[box['label']]))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    print(os.path.exists(FLAGS.label_map_path))
    dataset = FLAGS.data_dir
    # BOSCH
    examples = yaml.load(open(dataset, 'rb').read())

    print('Executing:', dataset)
    len_examples = len(examples)
    print("Loaded ", len(examples), "examples")

    for i in range(len(examples)):
        examples[i]['path'] = os.path.abspath(os.path.join(
            os.path.dirname(dataset), examples[i]['path']))

    counter = 0
    for example in examples:
        tf_example = create_tf_example(example, label_map_dict)
        writer.write(tf_example.SerializeToString())

        counter += 1
        print('Parsed:', counter)

    writer.close()


if __name__ == '__main__':
    tf.app.run()
