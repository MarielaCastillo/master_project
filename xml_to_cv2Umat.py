import os
import sys
import pandas as pd
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
    print(xml_file)
    print(xml_file.to_string())


    #for index, row in file_names.iterrows():
    #    print("index: ", index)
    #    print("row: ", row)
    print(file_names.shape)
    #print(file_names)
def get_objects(root):
    print(root)
    print(root[5][0].text)
    length = len(root.findall('object'))
    
    print(length)
    for i in range(5, 5+length):
        print("i",i)
        print(root[i][0].text)
        print(root[i][2][0].text)
        print(root[i][2][1].text)
        print(root[i][2][2].text)
        print(root[i][2][3].text)
        class_name = root[i][0].text
        xmin = root[i][2][0].text
        ymin = root[i][2][1].text
        xmax = root[i][2][2].text
        ymax = root[i][2][3].text

        


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
    get_objects(xml_root)
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