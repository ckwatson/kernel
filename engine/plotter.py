#
# plotter.py
#
"""Plot the solution that was generated by driver.py."""

# SciPy packages
import os, io#, mpld3
import numpy as np
from matplotlib.pyplot import figure, close
from pylab import subplot, plot, hold, legend, title, savefig, annotate
from matplotlib import rc

#from arrhenius import plot_arrhenius
from . import handy_functions as HANDY
from . import experiment_class, fileIO


# our colour 'array'
# allows for more concise loop code
colour = [0, 'b', 'g', 'r', 'c', 'm', 'y', 'k', '1', '0.75', '0.65', '0.55', '0.45', '0.35', '0.25', '0.25', '0.15']
if_SkipDrawingSpeciesWithZeroConcentrations = True
newSize = 100
sampler = lambda array: array[:,::array.shape[1]/newSize]
def fake_writer(plot):
    '''this function tricks matplotlib into writing figure into memory instead of an actual file on disk.'''
    buf = io.BytesIO()
    plot.savefig(buf, format="svg")
    buf.seek(0)
    data = buf.read()
    buf.close()
    return '<svg viewBox="0 0 2520 1584">'+data.decode("utf-8")[367:]

def sub_plots(Temperature, plottingDict, condition_fileName, solution_fileName, written_true_data = None, written_user_data = None):
    number_of_plots = len(plottingDict)
    # need to replace with logging
    stream.write("entered Plotter.sub_plots.\n        Attempting to plot " + str(number_of_plots) + " concentration profiles.")  
    # create the figure and determine the 'layout' of the subplots
    profiles = figure(figsize=(35, 22), dpi=80, facecolor='w', edgecolor='k', tight_layout = True) # figsize = (width, heigh)
    combined = figure(figsize=(35, 22), dpi=80, facecolor='w', edgecolor='k', tight_layout = True) # figsize = (width, heigh)
    dimensions = np.ceil(np.sqrt(number_of_plots))
    # need to replace with logging
    stream.write("        (a) load 2 set of data:")
    if written_true_data is not None:
        stream.write('            (i)  Data of true model, from memory cache.')
        true_data = written_true_data
    else: #we have to use data from file now...
        stream.write('            (i)  Data of true model, from "',condition_fileName + '_.dat','"...')
        true_data = fileIO.load_modelData(condition_fileName + '_.dat')
    if written_user_data is not None:
        if_userModel_failed = written_user_data is False
        if if_userModel_failed:
            stream.write('            (ii) Data of user model, but it failed according to memory cache.')
        else:
            stream.write('            (ii) Data of user model, from memory cache.')
            solu_data = written_user_data
    else: #we have to use data from file now...
        if_userModel_failed = os.path.isfile(solution_fileName+'_Failed') #heck whether this user model is okay, by checking the flag file.
        if if_userModel_failed:
            stream.write('            (ii) Data of user model, but it failed according to file stored.')
        else:
            stream.write('            (ii) Data of user model, from file stored.')
            solu_data = fileIO.load_modelData(solution_fileName + '_.dat')
        # need to replace with logging
    stream.write("        (b) draw the plots:")

    rc('font', size=22)

    sub_combined = combined.add_subplot(111, title ='Combined True Profile', xlabel='time', ylabel='Concentration')
    
    #pre-cache x-datapoints for 2 models:
    true_data_sampled = sampler(true_data)
    data_x_true = true_data_sampled[0,:]
    stream.write("            Lossy-compressing true_data by selecting only", newSize, 'items, which means a span of every',true_data.shape[1]/newSize,'items.\n            The true_data is compressed from',true_data.shape,'to',true_data_sampled.shape,'.')
    if not if_userModel_failed:
        solu_data_sampled = sampler(solu_data)
        data_x_solu = solu_data_sampled[0,:]
    stream.write("            Drawing curves for ", end = '')
    #now for every species to be plotted:
    for plot_info, (name, location) in enumerate(plottingDict.items(), start=1):
        stream.write(name, end = ', ')
        sub_individual = profiles.add_subplot(dimensions, dimensions, plot_info, title = 'Concentration Profile of '+name, xlabel='time', ylabel='[' + name + ']')
        # first true model:
        data_y_this = true_data_sampled[location+1,:]
        if not (if_SkipDrawingSpeciesWithZeroConcentrations and not any(y!=0 for y in data_y_this)):
            sub_individual.plot(data_x_true, data_y_this, colour[          1], label='True ' + '[' + name + ']', linestyle="-")
            sub_combined.  plot(data_x_true, data_y_this, colour[plot_info+2], label='True ' + '[' + name + ']', linestyle="-")
        # then user model:
        if not if_userModel_failed: 
            data_y_this = solu_data_sampled[location+1,:]
            if not (if_SkipDrawingSpeciesWithZeroConcentrations and not any(y!=0 for y in data_y_this)):
                sub_individual.plot(data_x_solu, data_y_this, colour[          2], label='User ' + '[' + name + ']', linestyle="--")
                sub_combined.plot  (data_x_solu, data_y_this, colour[plot_info+2], label='User ' + '[' + name + ']', linestyle="--")
        #sub_individual.legend()
    sub_combined.legend()
    stream.write(" done.\n        (c) Save plots to file: [Individual] ", end = '')
    #mpld3.save_html(profiles, profiles_filename+'.html')
    #mpld3.save_html(combined, combined_filename+'.html')
    #profiles.tight_layout()
    profiles = fake_writer(profiles)
    stream.write("[Combined] ", end = '')
    combined = fake_writer(combined)
    stream.write("Done.")
    return (profiles,combined)

if __name__ == '__main__':
    stream.write('Successfully loaded Plotter.py.')