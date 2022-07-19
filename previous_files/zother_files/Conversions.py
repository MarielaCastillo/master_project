import math
import numpy as np
import os

def conversionLatLongToXYZ(lat, lon):
        cosLat = math.cos(lat * math.pi / 180.0)
        sinLat = math.sin(lat * math.pi / 180.0)
        cosLon = math.cos(lon * math.pi / 180.0)
        sinLon = math.sin(lon * math.pi / 180.0)
        rad = 500.0
        x = rad * cosLat * cosLon
        y = rad * cosLat * sinLon
        z = rad * sinLat

        print(x, y, z)
        return np.array([x, y, z])

def conversionLatLongToXYAngle(lat, lon, angle):
        cosLat = math.cos(lat * math.pi / 180.0)
        sinLat = math.sin(lat * math.pi / 180.0)
        cosLon = math.cos(lon * math.pi / 180.0)
        sinLon = math.sin(lon * math.pi / 180.0)
        rad = 500.0
        rad = 6371.0 #Earth radius
        rad = 6.371
        x = rad * cosLat * cosLon
        y = rad * cosLat * sinLon
        #z = rad * sinLat

        print(x, y, angle)
        return np.array([x, y, angle])

def conversion2(lat, lon):
    earthRadius = 500
    earthRadius = 6.371
    x = earthRadius * math.cos(lat)*math.cos(lon)
    y = earthRadius * math.cos(lat)*math.sin(lon)
    z = earthRadius * math.sin(lat)

    print(x, y, z)
    return np.array([x, y, z])


def read_text_file(file_path):
    with open(file_path, 'r') as f:
        #print(file_path)
        #print(f.read())
        file = f.read()
        values = file.split(' ')
        return float(values[0]), float(values[1]), float(values[5])

def main():
    path = "/home/miw/Documents/Ubuntu stuff/KITTI/2011_09_26_drive_0001_extract/oxts/data"
    os.chdir(path)

    latarray = []
    lonarray = []

    values = []

    #a = np.array([[1, 2, 3], [4, 5, 6]])

    #newArray = np.append(a, [[50, 60, 70]], axis = 0)

    #print(newArray)
    '''
    b = np.empty((2,3))
    #b = np.append(b, [[50, 60, 70]], axis = 0)
    b[0,0] = 65450


    b = np.zeros((2,3))
    #b = np.append(b, [[50, 60, 70]], axis = 0)
    b[0,0] = 65450


    print(b)
    '''






    print("end")
    
    # iterate through all file
    for file in sorted(os.listdir()):
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{path}/{file}"
  
            # call read text file function
            lat, lon, angle = read_text_file(file_path)
            #print(lat, lon, angle)

            print(lat, lon, angle)

            #latarray.append(lat)
            #lonarray.append(lon)

        #xyangle1 = conversionLatLongToXYAngle(lat, lon, angle) #print
        #xyangle2 = conversion2(lat, lon) 

        #print(xyangle1)
    #print(math.pi/180)

    


    #print("end")

    
    

    
    '''    
    print("hola")
    xyz1 = conversionLatLongToXYZ(49.015026610861, 8.434354309633)
    conversion2(49.015026610861, 8.434354309633)
    print("hola2")
    xyz2 = conversionLatLongToXYZ(49.015025986508, 8.4343527579072)
    conversion2(49.015025986508, 8.4343527579072)
    print("hola3")

    result = xyz2 - xyz1
    print(result)
    '''
    




if __name__ == "__main__":
    main()       
    