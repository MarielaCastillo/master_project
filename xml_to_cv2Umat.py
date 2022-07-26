import os
import numpy as np
import pandas as pd
import PIL.Image
import torch
import xml.etree.ElementTree as ET 

import torchvision.transforms as transforms

dir_path = os.path.dirname(os.path.realpath(__file__))


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def get_tensor_painted(root, tensor, file):

    length = len(root.findall('object'))
    
    # print("number of objects: ", length)
    for i in range(5, 5+length):
        class_name = root[i][0].text
        xmin = int(root[i][2][0].text)
        ymin = int(root[i][2][1].text)
        xmax = int(root[i][2][2].text)
        ymax = int(root[i][2][3].text)

        if class_name == "car":
            colour = 1
        if class_name == "person":  # 00266
            colour = 2
        if class_name == "bicycle":  # 00266
            colour = 3
            # print("File with bicycle:", file)
        if class_name == "dog":  # 00266
            colour = 4
            # print("File with dog:", file)

        for i in range(xmin, xmax):
            for j in range(ymin, ymax):
                if i == 640:
                    print("file with ymax 640", file)
                    break
                if j == 512:
                    print("file with xmax 512", file)
                    break
                tensor[i,j] = colour
    
    return tensor


def get_images_train():
    # define folders and file
    dir_path3 = dir_path + '/' + 'align'
    txt_file = dir_path3 + '/' + 'align_train.txt'
    label_path=dir_path3 + '/' + 'Annotations'

    # get path of the xml file
    file_names = pd.read_csv(txt_file, sep=" ", header=None)
    lbl_str = ''.join(label_path)  #change tuple to str

    for index, row in file_names.iterrows():
        file_name = file_names.loc[index, 0].replace(" ", "")
        xml_path = os.path.join(lbl_str, file_name+".xml")

        tree = ET.parse(xml_path) 
        xml_root = tree.getroot() 

        val = row.values
        val= str(val)

        file_id = val.replace("_PreviewData", "")
        file_id = file_id.replace("['", "")
        file_id = file_id.replace("']", "")

        file_id_jpg = file_id + "_labels.jpg"

        # ------------
        tensor = torch.zeros(640, 512)
        new_tensor = get_tensor_painted(xml_root, tensor, file_id_jpg)
        new_tensor = torch.transpose(new_tensor, 0,1)
        image = tensor_to_image(new_tensor)

        image.save("labels_ss/"+file_id)

        # if index == 0:
        #    break

def get_images_val():
    # define folders and file
    dir_path3 = dir_path + '/' + 'align'
    txt_file = dir_path3 + '/' + 'align_validation.txt'
    label_path=dir_path3 + '/' + 'Annotations'

    # get path of the xml file
    file_names = pd.read_csv(txt_file, sep=" ", header=None)
    lbl_str = ''.join(label_path)  #change tuple to str

    for index, row in file_names.iterrows():
        file_name = file_names.loc[index, 0].replace(" ", "")
        xml_path = os.path.join(lbl_str, file_name+".xml")

        tree = ET.parse(xml_path) 
        xml_root = tree.getroot() 

        val = row.values
        val= str(val)

        file_id = val.replace("_PreviewData", "")
        file_id = file_id.replace("['", "")
        file_id = file_id.replace("']", "")

        file_id_jpg = file_id + "_labels.jpg"

        # ------------
        tensor = torch.zeros(640, 512)
        new_tensor = get_tensor_painted(xml_root, tensor, file_id_jpg)
        new_tensor = torch.transpose(new_tensor, 0,1)
        image = tensor_to_image(new_tensor)

        image.save("labels__images_val/"+file_id)

        # if index == 0:
        #    break

def get_npys():
    # define folders and file
    dir_path3 = dir_path + '/' + 'align'
    txt_file = dir_path3 + '/' + 'align_train.txt'
    label_path=dir_path3 + '/' + 'Annotations'

    # get path of the xml file
    file_names = pd.read_csv(txt_file, sep=" ", header=None)
    lbl_str = ''.join(label_path)  #change tuple to str

    for index, row in file_names.iterrows():
        file_name = file_names.loc[index, 0].replace(" ", "")
        xml_path = os.path.join(lbl_str, file_name+".xml")

        tree = ET.parse(xml_path) 
        xml_root = tree.getroot() 

        val = row.values
        val= str(val)

        file_id = val.replace("_PreviewData", "")
        file_id = file_id.replace("['", "")
        file_id = file_id.replace("']", "")

        file_id_jpg = file_id + "_labels.jpg"

        # ------------
        tensor = torch.zeros(640, 512)
        new_tensor = get_tensor_painted(xml_root, tensor, file_id_jpg)
        new_tensor = torch.transpose(new_tensor, 0,1)
        image = tensor_to_image(new_tensor)

        np.save("labels_npy/"+file_id, new_tensor)

def get_npys_val():
    # define folders and file
    dir_path3 = dir_path + '/' + 'align'
    txt_file = dir_path3 + '/' + 'align_validation.txt'
    label_path=dir_path3 + '/' + 'Annotations'

    # get path of the xml file
    file_names = pd.read_csv(txt_file, sep=" ", header=None)
    lbl_str = ''.join(label_path)  #change tuple to str

    for index, row in file_names.iterrows():
        file_name = file_names.loc[index, 0].replace(" ", "")
        xml_path = os.path.join(lbl_str, file_name+".xml")

        tree = ET.parse(xml_path) 
        xml_root = tree.getroot() 

        val = row.values
        val= str(val)

        file_id = val.replace("_PreviewData", "")
        file_id = file_id.replace("['", "")
        file_id = file_id.replace("']", "")

        file_id_jpg = file_id + "_labels.jpg"

        # ------------
        tensor = torch.zeros(640, 512)
        new_tensor = get_tensor_painted(xml_root, tensor, file_id_jpg)
        new_tensor = torch.transpose(new_tensor, 0,1)
        image = tensor_to_image(new_tensor)

        np.save("labels_npy_val/"+file_id, new_tensor)
        


def main():
    #image = get_images()
    ## Define a transform to convert the image to tensor
    #transform = transforms.ToTensor()

    ## Convert the image to PyTorch tensor
    #tensor = transform(image)

    #get_images()
    get_npys_val()
    
    print("yay")


if __name__ == "__main__":
    main()
