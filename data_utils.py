from PIL import Image
import glob
import numpy as np
import os



def load_dir():
    directory = "./Images"

    capitals_list = os.listdir(directory)

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    cur_capital_class = 0
    for cur_capital in capitals_list:
        
        cur_capital_directory = "./Images/" + cur_capital

        cur_capital_image_list = os.listdir(cur_capital_directory)

        for imgIdx in range(0, len(cur_capital_image_list)):
            filename = cur_capital_directory + "/" + cur_capital_image_list[imgIdx]
            if (not os.path.isfile(filename)):
                continue
            im=Image.open(filename)
            cur_img_np_array = np.array(im)
            red_channel = cur_img_np_array[:, :, 0]
            green_channel = cur_img_np_array[:, :, 1]
            blue_channel = cur_img_np_array[:, :, 2]
            img_channels = [red_channel, green_channel, blue_channel]
            if imgIdx < (len(cur_capital_directory) * 3 / 4):
                x_train.append(img_channels)
                y_train.append(cur_capital_class)
            else:
                x_test.append(img_channels)
                y_test.append(cur_capital_class)
        cur_capital_class += 1
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test
