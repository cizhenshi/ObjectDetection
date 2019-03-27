from xml.dom.minidom import Document
import cv2
import os
import glob
import shutil
import numpy as np

from xml.dom.minidom import Document
import cv2
import os
import glob
import shutil
import numpy as np

def generate_xml(name, classes, XMIN, YMIN, XMAX, YMAX, img_size, class_sets, doncateothers=True):
    doc = Document()

    def append_xml_node_attr(child, parent=None, text=None):
        ele = doc.createElement(child)
        if not text is None:
            text_node = doc.createTextNode(text)
            ele.appendChild(text_node)
        parent = doc if parent is None else parent
        parent.appendChild(ele)
        return ele

    img_name = name + '.jpg'
    # create header
    annotation = append_xml_node_attr('annotation')
    append_xml_node_attr('folder', parent=annotation, text='text')
    append_xml_node_attr('filename', parent=annotation, text=img_name)
    source = append_xml_node_attr('source', parent=annotation)
    append_xml_node_attr('database', parent=source, text='coco_text_database')
    append_xml_node_attr('annotation', parent=source, text='text')
    append_xml_node_attr('image', parent=source, text='text')
    append_xml_node_attr('flickrid', parent=source, text='000000')
    owner = append_xml_node_attr('owner', parent=annotation)
    append_xml_node_attr('name', parent=owner, text='ms')
    size = append_xml_node_attr('size', annotation)
    append_xml_node_attr('width', size, str(img_size[1]))
    append_xml_node_attr('height', size, str(img_size[0]))
    append_xml_node_attr('depth', size, str(img_size[2]))
    append_xml_node_attr('segmented', parent=annotation, text='0')

    # create objects
    objs = []
    for index in range(0, len(classes)):
        cls = classes[index]
        xmin = float(XMIN[index])
        ymin = float(YMIN[index])
        xmax = float(XMAX[index])
        ymax = float(YMAX[index])
        if not doncateothers and cls not in class_sets:
            continue
        cls = 'dontcare' if cls not in class_sets else cls
        if cls == 'dontcare':
            continue
        obj = append_xml_node_attr('object', parent=annotation)
        occlusion = int(0)
        loc = np.clip(np.asarray([xmin, ymin, xmax, ymax], dtype=float),1,2047)
        xmin, ymin, xmax, ymax = loc[0], loc[1], loc[2], loc[3]
        if xmax < xmin:
            temp = xmax
            xmax = xmin
            xmin = temp
        if ymax < ymin:
            temp = ymax
            ymax = ymin
            ymin = temp

        truncation = float(0)
#         difficult = 1 if _is_hard(cls, truncation, occlusion, xmin, ymin, xmax, ymax) else 0
        difficult = 0
        truncted = 0 if truncation < 0.5 else 1

        append_xml_node_attr('name', parent=obj, text=cls)
        append_xml_node_attr('pose', parent=obj, text='none')
        append_xml_node_attr('truncated', parent=obj, text=str(truncted))
        append_xml_node_attr('difficult', parent=obj, text=str(int(difficult)))
        bb = append_xml_node_attr('bndbox', parent=obj)
        append_xml_node_attr('xmin', parent=bb, text=str(xmin))
        append_xml_node_attr('ymin', parent=bb, text=str(ymin))
        append_xml_node_attr('xmax', parent=bb, text=str(xmax))
        append_xml_node_attr('ymax', parent=bb, text=str(ymax))

        o = {'class': cls, 'box': np.asarray([xmin, ymin, xmax, ymax], dtype=float), \
             'truncation': truncation, 'difficult': difficult, 'occlusion': occlusion}
        objs.append(o)

    return doc, objs


def _is_hard(cls, truncation, occlusion, x1, y1, x2, y2):
    hard = False
    if y2 - y1 < 25 and occlusion >= 2:
        hard = True
        return hard
    if occlusion >= 3:
        hard = True
        return hard
    if truncation > 0.8:
        hard = True
        return hard
    return hard


def build_voc_dirs(outdir):
    mkdir = lambda dir: os.makedirs(dir) if not os.path.exists(dir) else None
    mkdir(outdir)
    mkdir(os.path.join(outdir, 'Annotations'))
    mkdir(os.path.join(outdir, 'ImageSets'))
    mkdir(os.path.join(outdir, 'ImageSets', 'Layout'))
    mkdir(os.path.join(outdir, 'ImageSets', 'Main'))
    mkdir(os.path.join(outdir, 'ImageSets', 'Segmentation'))
    mkdir(os.path.join(outdir, 'JPEGImages'))
    mkdir(os.path.join(outdir, 'SegmentationClass'))
    mkdir(os.path.join(outdir, 'SegmentationObject'))
    return os.path.join(outdir, 'Annotations'), os.path.join(outdir, 'JPEGImages'), os.path.join(outdir, 'ImageSets',
                                                                                                 'Main')


if __name__ == '__main__':
    _outdir = 'TT100KVOC/VOC2007'
    _draw = bool(0)
    _dest_label_dir, _dest_img_dir, _dest_set_dir = build_voc_dirs(_outdir)
    print(_dest_label_dir, _dest_img_dir, _dest_set_dir)
    _doncateothers = bool(1)
    for phase in ['train', 'test']:
        label = "./data/"+ phase + ".txt"
        fp = open(label, 'r')
        class_sets = ('pn', 'ph5', 'pl80', 'pl120', 'w32', 'pl50', 'wo', 'pl30', 'pl60', 'pm55', 'pl20', 'pm30', 'p26', 'w13', 'p5', 'po', 'ph4', 'pne', 'p12', 'w55', 'pl70', 'io', 'i2', 'ph4.5', 'pl40', 'p6', 'pg', 'pm20', 'pl100', 'p27', 'pr40', 'w57', 'pl5', 'p11', 'il80', 'p23', 'w59', 'il100', 'i5', 'p3', 'i4', 'il60', 'p19', 'ip', 'p10', 'dontcare')
        class_sets_dict = dict((k, i) for i, k in enumerate(class_sets))
        allclasses = {}
        fs = [open(os.path.join(_dest_set_dir, cls + '_' + phase + '.txt'), 'w') for cls in class_sets]
        ftrain = open(os.path.join(_dest_set_dir, phase + '.txt'), 'w')
        for line in tqdm(fp):
            line = line.strip()
            items = line.split()
            image_dir = items[0]
            stem, ext = os.path.splitext(image_dir.split('/')[-1])
            classes = items[1::5]
            XMIN = items[2::5]
            YMIN = items[3::5]
            XMAX = items[4::5]
            YMAX = items[5::5]
            img = cv2.imread(image_dir)
            img_size = img.shape
            assert(len(XMIN) == len(classes), 'check format of input, the number of location is not the same!!')
            doc, objs = generate_xml(stem, classes, XMIN, YMIN, XMAX, YMAX, img_size, class_sets=class_sets, doncateothers=_doncateothers)
            cv2.imwrite(os.path.join(_dest_img_dir, stem + '.jpg'), img)
            xmlfile = os.path.join(_dest_label_dir, stem + '.xml')
            with open(xmlfile, 'w') as f:
                f.write(doc.toprettyxml(indent='	'))

            ftrain.writelines(stem + '\n')

            cls_in_image = set([o['class'] for o in objs])

            for obj in objs:
                cls = obj['class']
                allclasses[cls] = 0 \
                    if not cls in list(allclasses.keys()) else allclasses[cls] + 1

            for cls in cls_in_image:
                if cls in class_sets:
                    fs[class_sets_dict[cls]].writelines(stem + ' 1\n')
            for cls in class_sets:
                if cls not in cls_in_image:
                    fs[class_sets_dict[cls]].writelines(stem + ' -1\n')


        (f.close() for f in fs)
        ftrain.close()

        print('~~~~~~~~~~~~~~~~~~~')
        print(allclasses)
        print('~~~~~~~~~~~~~~~~~~~')
        shutil.copyfile(os.path.join(_dest_set_dir, 'train.txt'), os.path.join(_dest_set_dir, 'val.txt'))
        shutil.copyfile(os.path.join(_dest_set_dir, 'train.txt'), os.path.join(_dest_set_dir, 'trainval.txt'))
        for cls in class_sets:
            shutil.copyfile(os.path.join(_dest_set_dir, cls + '_train.txt'),
                            os.path.join(_dest_set_dir, cls + '_trainval.txt'))
            shutil.copyfile(os.path.join(_dest_set_dir, cls + '_train.txt'),
                            os.path.join(_dest_set_dir, cls + '_val.txt'))
