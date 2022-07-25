import os
import numpy as np
import pandas as pd
import itertools as it
import PIL.Image
import torch
import xml.etree.ElementTree as ET 


dir_path = os.path.dirname(os.path.realpath(__file__))

def main2():
    # define folders and file
    dir_path3 = dir_path + '/' + 'align'
    txt_file = dir_path3 + '/' + 'align_train.txt'
    label_path=dir_path3 + '/' + 'Annotations'

    file_names = pd.read_csv(txt_file, sep=" ", header=None)
    print(type(file_names))
    print("name of file",file_names.loc[0, 0])

    lbl_str = ''.join(label_path)

    file_name = file_names.loc[0, 0].replace(" ", "")

    xml_path = os.path.join(lbl_str, file_name+".xml")
    print("xml_path", xml_path)
    xml_file = pd.read_xml(xml_path)


    #for index, row in file_names.iterrows():
    #    print("index: ", index)
    #    print("row: ", row)
    print(file_names.shape)
    #print(file_names)

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def paint(tensor, class_name, xmin, xmax, ymin, ymax):
    print("hi")
    for i in it.chain(range(30, 52), range(1, 18)):
        print(i)




def get_tensor_painted(root, tensor):
    print(root)
    print(root[5][0].text)
    length = len(root.findall('object'))
    
    print("number of objects: ", length)
    for i in range(5, 5+length):
        print("i",i)
        '''
        print(root[i][0].text)
        print(root[i][2][0].text)
        print(root[i][2][1].text)
        print(root[i][2][2].text)
        print(root[i][2][3].text)
        '''
        class_name = root[i][0].text
        xmin = int(root[i][2][0].text)
        ymin = int(root[i][2][1].text)
        xmax = int(root[i][2][2].text)
        ymax = int(root[i][2][3].text)

        if class_name == "car":
            colour = 1
        if class_name == "person":
            colour = 2
        if class_name == "bicycle":
            colour = 3

        for i in range(xmin, xmax):
            for j in range(ymin, ymax):
                tensor[i,j] = colour
    
    return tensor

        

        

        


def main():
    # define folders and file
    dir_path3 = dir_path + '/' + 'align'
    txt_file = dir_path3 + '/' + 'align_train.txt'
    label_path=dir_path3 + '/' + 'Annotations'

    # get path of the xml file
    file_names = pd.read_csv(txt_file, sep=" ", header=None)
    lbl_str = ''.join(label_path) #change tuple to str
    file_name = file_names.loc[0, 0].replace(" ", "")
    print("file_name", file_name)
    xml_path = os.path.join(lbl_str, file_name+".xml")
    print("xml_path", xml_path)

    tree = ET.parse(xml_path) 
    xml_root = tree.getroot() 



    # ------------
    tensor = torch.zeros(640, 512)
    # tensor[:,1:50] = 1
    # print(tensor)

    # ------------

    new_tensor = get_tensor_painted(xml_root, tensor)
    new_tensor = torch.transpose(new_tensor, 0,1)
    image = tensor_to_image(new_tensor)

    im1 = image.save("imageasdf5.jpg")



    print("yay")
    
    '''
    print(root)
    print(root[0].attrib) 
    print(root[2][0].text) 
    print(root[5][0].text)  # <- first object
    #print(tree.shape)
    print(sys.getsizeof(tree))
    length = len(root.findall('object'))
    print(length)
    '''





    



if __name__ == "__main__":
    main()