import pandas as pd, numpy as np
from xml.etree.ElementTree import Element, SubElement, ElementTree


# label_data : csv file


#landmark_name = ['Glabella', 'R3', 'Nasion']
landmark_name = [
        'Nasion',
        'Sella',
        'Porion',
        'Orbitale',
        'Pterygoid',
        'Basion',
        'ANS',
        'PNS',
        'A point',
        'B point',
        'Protuberance menti',
        'Pogonion',
        'Menton',
        'Go-1 Corpuus left',
        'Go-2 Ramus down',
        'Articulare',
        'R3',
        'R1',
        'Maxilla 1 crown',
        'Maxilla 1 root',
        'Mandible 1 crown',
        'Mandible 1 root',
        'Maxilla 6 distal',
        'Maxilla 6 root',
        'Mandible 6 distal',
        'Mandible 6 root',
        'Glabella',
        'Soft tissue naison',
        'Dorsum of nose',
        'pronasale',
        'Colunella',
        'Subnasalle',
        'Soft tissue A point',
        'Labrale superius',
        'Upper lip',
        'Stms',
        'Upper embrasure',
        'Lower embrasure',
        'Stmi',
        'Lower lip',
        'Labial Inferius',
        'Soft tissue b point',
        'Soft tissue pogonion',
        'Soft tissue Gnathion',
        'Soft tissue Menton',
        'Cervical point',  
    ]



def xml_writer(label_data):
    
    len_file = len(label_data['ID'])

    bbox_size = 32
        
    for i in range(len_file):
        root = Element("annotation")

        element1 = Element("filename")
        root.append(element1)
        element1.text = label_data.iloc[i]['ID']


        element2 = Element("size")
        root.append(element2)
        
        sub_element2 = SubElement(element2, "width")
        sub_element2.text = '512'
        sub_element2 = SubElement(element2, "height")
        sub_element2.text = '512'
        sub_element2 = SubElement(element2, "depth")
        sub_element2.text = '3'

        

        for j in range( len(landmark_name) ):
            element3 = Element("object")
            root.append(element3)
            sub_element3 = SubElement(element3, "name")
            sub_element3.text = landmark_name[j]
            
            x_loc = label_data.iloc[i][landmark_name[j]+'_x']
            y_loc = label_data.iloc[i][landmark_name[j]+'_y']
            
            
            tmp = np.array([x_loc*512.-bbox_size, x_loc*512.+bbox_size, y_loc*512.-bbox_size, y_loc*512.+bbox_size]).astype(int)


            sub_element4 = SubElement(element3, "bndbox")

            sub_element5 = SubElement(sub_element4, "xmin")
            sub_element5.text = str(tmp[0])

            sub_element6 = SubElement(sub_element4, "ymin")
            sub_element6.text = str(tmp[2])

            sub_element7 = SubElement(sub_element4, "xmax")
            sub_element7.text = str(tmp[1])

            sub_element8 = SubElement(sub_element4, "ymax")
            sub_element8.text = str(tmp[3])
            
        
        tree = ElementTree(root)
        
        #i_2 = '{0:04d}'.format(i)
        i_2 = label_data['ID'][i]
        fileName = f"./Annotations/{i_2}.xml"
        
        with open(fileName, "wb") as file:
            tree.write(file, encoding='utf-8', xml_declaration=True)