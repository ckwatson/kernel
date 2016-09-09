import numpy as np

if_reallyWrite = True

def save_figure(data_to_write, file_name = ''):
    '''This function saves plot to file'''
    if if_reallyWrite:
        with open(file_name, 'w') as outfile:
            outfile.write(data_to_write)

def save_modelData(data_to_write, file_name):
    '''The new "write_ODE".'''
    if if_reallyWrite:
        if data_to_write is False: # works as "write_failed_userData(temperature, data_file_name)"
            open(file_name+"_Failed", 'a').close()
        else:
            with open(file_name+'_.dat', 'wb') as outfile:
                np.save(outfile, data_to_write)

load_modelData = np.load