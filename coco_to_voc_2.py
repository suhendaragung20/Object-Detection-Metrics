
# Usage
# python coco_to_voc.py --anno data_coco/part_1 --image data_image --voc data_voc

import xml.etree.ElementTree as ET
from xml.dom import minidom
#from utils import plot_cv

import shutil

import os
import numpy as np
import cv2


import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--anno", type=str,
    help="path to input annotations")
ap.add_argument("-i", "--image", type=str,
    help="path to input images")
ap.add_argument("-v", "--voc", type=str,
    help="path to output voc")
args = vars(ap.parse_args())


class cleansing_data:

    def __init__(self, args):
        self.aa = 1
        self.annotations_folder = args["anno"]
        self.images_folder = args["image"]
        self.fix_folder = args["voc"]
        self.image_files = os.listdir(self.images_folder)

        self.num_fail = 0
        self.num_data = 0

        self.list_voc = []


    def clear_image(self):
        if os.path.exists(self.fix_folder) == True:
            shutil.rmtree(self.fix_folder)

        if os.path.exists(self.fix_folder) == False:
            os.makedirs(self.fix_folder)


        print('delete all file')


    def get_num_images(self):
        return len(self.image_files)


    def getText(self, nodelist):
        # Iterate all Nodes aggregate TEXT_NODE
        rc = []
        for node in nodelist:
            if node.nodeType == node.TEXT_NODE:
                rc.append(node.data)
            else:
                # Recursive
                rc.append(getText(node.childNodes))
        return ''.join(rc)


    def prettify(self, elem):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")


    def write_to_xml(self, augmented, class_names, target_path, image, image_name, image_path):
        # -------------------------------------------------------------------
        # Writing new XML file 

        image_h, image_w = image.shape[:2]

        xannotation = ET.Element('annotation')

        xfolder = ET.SubElement(xannotation, 'folder')
        xfolder.text = 'None'

        xfilename = ET.SubElement(xannotation, 'filename')
        xfilename.text = image_name

        xpath = ET.SubElement(xannotation, 'path')
        xpath.text = image_path

        xsource = ET.SubElement(xannotation, 'source')

        xdatabase = ET.SubElement(xsource, 'database')
        xdatabase.text = 'Unknown'

        xsize = ET.SubElement(xannotation, 'size')

        xwidth = ET.SubElement(xsize, 'width')
        xwidth.text = str(image_w)

        xheight = ET.SubElement(xsize, 'height')
        xheight.text = str(image_h)

        xdepth = ET.SubElement(xsize, 'depth')
        xdepth.text = str(3)

        xsegmented = ET.SubElement(xannotation, 'segmented')
        xsegmented = str(0)

        xobjects = []

        for box, idx in zip(augmented["bboxes"], augmented["category_id"]):

            (xmin, ymin, wBox, hBox) = box

            xobject = ET.SubElement(xannotation, 'object')

            xname = ET.SubElement(xobject, 'name')
            xname.text = class_names[idx]

            xpose = ET.SubElement(xobject, 'pose')
            xpose.text = 'Unspecified'

            xtruncated = ET.SubElement(xobject, 'truncated')
            xtruncated.text  = str(0)

            xdifficult = ET.SubElement(xobject, 'difficult')
            xdifficult.text  = str(0)

            xbndbox = ET.SubElement(xobject, 'bndbox')

            xxmin = ET.SubElement(xbndbox, 'xmin')
            xxmin.text = str(int(xmin))

            xymin = ET.SubElement(xbndbox, 'ymin')
            xymin.text = str(int(ymin))

            xxmax = ET.SubElement(xbndbox, 'xmax')
            xxmax.text = str(int(xmin + wBox))

            xymax = ET.SubElement(xbndbox, 'ymax')
            xymax.text = str(int(ymin + hBox))

        # -------------------------------------------------------------------

        #print(prettify(xannotation))

        myfile = open(target_path, "w")
        myfile.write(self.prettify(xannotation))


    def write_to_voc(self):
        voc_string_txt = ''
        for voc in self.list_voc:
            (bboxes, category_id, class_names, (h, w), target_save_image_path) = voc
            # voc_string_txt = voc_string_txt + '' + target_save_image_path
            for box, idx in zip(bboxes, category_id):
                (xmin, ymin, wBox, hBox) = box
                xmax = xmin + wBox
                ymax = ymin + hBox
                if xmax >= w:
                    xmax = w - 1
                if ymax >= h:
                    ymax = h - 1

                name = class_names[idx]
                if name == 'plate':
                    id_voc = 1
                elif name == '1':
                    id_voc = 2
                elif name == '2':
                    id_voc = 3
                elif name == '3':
                    id_voc = 4
                elif name == '4':
                    id_voc = 5
                elif name == '5':
                    id_voc = 6
                elif name == '6':
                    id_voc = 7
                elif name == '7':
                    id_voc = 8
                elif name == '8':
                    id_voc = 9
                elif name == '9':
                    id_voc = 10
                elif name == '0':
                    id_voc = 11
                elif name == 'A':
                    id_voc = 12
                elif name == 'B':
                    id_voc = 13
                elif name == 'C':
                    id_voc = 14
                elif name == 'D':
                    id_voc = 15
                elif name == 'E':
                    id_voc = 16
                elif name == 'F':
                    id_voc = 17
                elif name == 'G':
                    id_voc = 18
                elif name == 'H':
                    id_voc = 19
                elif name == 'I':
                    id_voc = 20
                elif name == 'J':
                    id_voc = 21
                elif name == 'K':
                    id_voc = 22
                elif name == 'L':
                    id_voc = 23
                elif name == 'M':
                    id_voc = 24
                elif name == 'N':
                    id_voc = 25
                elif name == 'O':
                    id_voc = 26
                elif name == 'P':
                    id_voc = 27
                elif name == 'Q':
                    id_voc = 28
                elif name == 'R':
                    id_voc = 29
                elif name == 'S':
                    id_voc = 30
                elif name == 'T':
                    id_voc = 31
                elif name == 'U':
                    id_voc = 32
                elif name == 'V':
                    id_voc = 33
                elif name == 'W':
                    id_voc = 34
                elif name == 'X':
                    id_voc = 35
                elif name == 'Y':
                    id_voc = 36
                elif name == 'Z':
                    id_voc = 37

                voc_string_txt = voc_string_txt + '' + str(name) + ' ' \
                                                        + str(int(xmin)) + ' ' \
                										+ str(int(ymin)) + ' ' \
            											+ str(int(xmax)) + ' ' \
            											+ str(int(ymax)) + '' 

                voc_string_txt = voc_string_txt + '\n'

            with open('groundtruths/' + target_save_image_path + '.txt', 'w') as f:  # write increment idx last send
                f.write("%s\n" % voc_string_txt)
            voc_string_txt = ''



    def do_fixing(self, image, xmldoc):
        nodelistName = xmldoc.getElementsByTagName('name')
        nodelistXmin = xmldoc.getElementsByTagName('xmin')
        nodelistYmin = xmldoc.getElementsByTagName('ymin')
        nodelistXmax = xmldoc.getElementsByTagName('xmax')
        nodelistYmax = xmldoc.getElementsByTagName('ymax')

        boxes = []
        idx_classes = []

        class_names = []

        # Iterate <text ..>...</text> Node List
        for nodename, nodexmin, nodeymin, nodexmax, nodeymax in zip(nodelistName, nodelistXmin, nodelistYmin, nodelistXmax, nodelistYmax):
            name = self.getText(nodename.childNodes)
            xmin = int(self.getText(nodexmin.childNodes))
            ymin = int(self.getText(nodeymin.childNodes))
            xmax = int(self.getText(nodexmax.childNodes))
            ymax = int(self.getText(nodeymax.childNodes))
            wBox = xmax - xmin
            hBox = ymax - ymin
            #print(name)
            #print(xmin, ymin, xmax, ymax)
            boxes.append((xmin, ymin, wBox, hBox))

            if (name in class_names) == False:
                class_names.append(name)

            idx_classes.append(class_names.index(name))


        annotations = {'image': image, 'bboxes': boxes, 'category_id': idx_classes}

        category_id_to_name = {}
        i = 0
        for name in class_names:
            category_id_to_name[i] = name
            i += 1

        return annotations, class_names




    def fixing(self, idx_data):

        file_image = self.image_files[idx_data]

        image = cv2.imread(self.images_folder+'/'+file_image)
        h, w = image.shape[:2]

        filename_without_extension = os.path.splitext(file_image)[0]

        print(self.annotations_folder + '/' + filename_without_extension + '.xml')

        try:
            xmldoc = minidom.parse(self.annotations_folder + '/' + filename_without_extension + '.xml')
        except:
            # print("miss xml file")
            self.num_fail += 1 
            return 0



        # Simpan image hasil augment flip Horizontal
        anno, class_names = self.do_fixing(image, xmldoc)



        self.num_data += 1

        target_image_name = "image_" + str(self.num_data) + '.jpg' 
        target_save_image_path = self.fix_folder + '/' + target_image_name

        cv2.imwrite(target_save_image_path, anno['image'])

        target_image_name = "image_" + str(self.num_data)
        target_save_image_path = target_image_name

        h, w = anno['image'].shape[:2]

        print(anno["bboxes"])

        # self.write_to_xml(anno, class_names, target_save_xml_path, anno['image'], target_image_name, target_save_image_path)
        self.list_voc.append((anno["bboxes"], anno["category_id"], class_names, (h, w), target_save_image_path))

        return 1




def main(args):
    cln = cleansing_data(args)
    cln.clear_image()
    for i_train in range(cln.get_num_images()):
        print("train [" + str(i_train+1) + '/' + str(cln.get_num_images()) + "]")
        #try:
        cln.fixing(i_train)
        #except:
        #	pass

    cln.write_to_voc()

main(args)