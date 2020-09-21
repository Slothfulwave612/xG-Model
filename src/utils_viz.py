'''
__author__: slothfulwave612
'''

## required packages/modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib import colors
import seaborn as sns
import dataframe_image as dfi
import utils_io as uio

class Pitch:
    '''
    class to create a pitch-map.
    
    Pitch Dimensions: 104x64 (in meters)
    '''
    
    def __init__(self, line_color='#000000', pitch_color='#FFFFFF', orientation='horizontal', figsize_x=10.4, figsize_y=6.8):
        '''
        Function to initialize the class object.
        
        Arguments:
            self -- represents the object of the class.
            line_color -- str, color of the lines. Default: '#000000'
            pitch_color -- str, color of the pitch-map. Default: '#FFFFFF'
            orientation -- str, pitch orientation 'horizontal' or 'vertical'. Default: 'horizontal'
        '''
        
        self.line_color = line_color
        self.pitch_color = pitch_color
        self.orientation = orientation
        
        ## figsize
        self.figsize_x = figsize_x              ## x-axis length
        self.figsize_y = figsize_y              ## y-axis length
        
        ## pitch outfield lines
        self.pitch_dims = [
            [0, 104, 104, 0, 0],                ## x-axis
            [0, 0, 68, 68, 0]                   ## y-axis
        ]
        
        self.dims_x = [
            [104, 87.5, 87.5, 104],             ## right-side penalty box(x-axis)
            [0, 16.5, 16.5, 0],                 ## left-side penalty box(x-axis)
            [104, 104.2, 104.2, 104],           ## right-side goal post(x-axis)
            [0, -0.2, -0.2, 0],                 ## left-side goal post(x-axis)
            [104, 99.5, 99.5, 104],             ## right-side 6-yard-box(x-axis)
            [0, 4.5, 4.5, 0]                    ## left-side 6-yard-box(x-axis)
        ]
        
        self.dims_y = [
            [13.84, 13.84, 54.16, 54.16],       ## right-side penalty box(y-axis)
            [13.84, 13.84, 54.16, 54.16],       ## left-side penalty box(y-axis)
            [30.34, 30.34, 37.66, 37.66],       ## right-side goal post(y-axis)
            [30.34, 30.34, 37.66, 37.66],       ## left-side goal post(y-axis)
            [24.84, 24.84, 43.16, 43.16],       ## right-side 6-yard-box(y-axis)
            [24.84, 24.84, 43.16, 43.16]        ## left-side 6-yard-box(y-axis)
        ]
        
        self.half_x = [52, 52]                  ## halfway line x-axis
        self.half_y = [0, 68]                   ## halfway line y-axis
        
        self.scatter_x = [93, 10.5, 52]         ## penalty and kick off spot(x-axis)
        self.scatter_y = [34, 34, 34]           ## penalty and kick off spot(y-axis)
        
        self.rect_x = [87.5, 0]                 ## rectangle (x-axis)
        self.rect_y = [20, 20]                  ## rectange (y-axis)
        
        self.rect = [16.5, 24]                  ## rectangle for drawing arc
        
    def create_pitch(self, zorder_line=3, fill_rect=None, length=None, bredth=None):
        '''
        Function to create pitch-map.
        
        Arguments:
            self -- represents the object of the class.
            zorder_line -- int, zorder value for pitch-lines.
            fill_rect -- list of x and y coordinates for the rectangle.
            length -- float, length of the rectangle.
            bredth -- float, bredth of the rectangle.
        
        Returns:
            fig, ax -- figure and axis object.
        '''
        
        if self.orientation == 'horizontal':
            ## figsize
            sx, sy = self.figsize_x, self.figsize_y
            
            ## x and y coordinates for pitch outline
            pitch_dims_x = self.pitch_dims[0]
            pitch_dims_y = self.pitch_dims[1]
            
            ## x and y coordinates for halfway line
            half_x = self.half_x
            half_y = self.half_y
            
            ## x and y coordinate list
            x_coord = self.dims_x
            y_coord = self.dims_y
            
            ## x and y coordinate for spots
            spot_x = self.scatter_x
            spot_y = self.scatter_y
            
            ## x and y coordinate for rectangle
            r_x = self.rect_x
            r_y = self.rect_y
            
            ## rectangle for horizontal orientation
            r_o = self.rect
                
            if length == None:
                length = 124
            
            if bredth == None:
                bredth = 95
            
        elif self.orientation == 'vertical':
            ## figsize
            sx, sy = self.figsize_y, self.figsize_x
            
            ## x and y coordinates for pitch outline
            pitch_dims_x = self.pitch_dims[1]
            pitch_dims_y = self.pitch_dims[0]
            
            ## x and y coordinates for halfway line
            half_x = self.half_y
            half_y = self.half_x
            
            ## x and y coordinate list
            x_coord = self.dims_y
            y_coord = self.dims_x
        
            ## x and y coordinate for spots
            spot_x = self.scatter_y
            spot_y = self.scatter_x 
            
            ## x and y coordinate for rectangle
            r_x = self.rect_y
            r_y = self.rect_x
            
            ## rectangle for vertical orientation
            r_o = self.rect[::-1]
                
            if length == None:
                length = 95
            
            if bredth == None:
                bredth = 124
        
        else:
            raise Exception('Orientation not understood!!!')
        
        if fill_rect == None:
            fill_rect = [-5, -10]
        
        ## create subplot
        fig, ax = plt.subplots(figsize=(sx, sy), facecolor=self.pitch_color)
        ax.set_facecolor(self.pitch_color)
    
        ## plot outfield lines
        ax.plot(pitch_dims_x, pitch_dims_y, color=self.line_color, zorder=zorder_line)
        
        ## plot right side penalty box
        ax.plot(x_coord[0], y_coord[0], color=self.line_color, zorder=zorder_line)
        
        ## plot left side penalty box
        ax.plot(x_coord[1], y_coord[1], color=self.line_color, zorder=zorder_line)
        
        ## plot right side goal post
        ax.plot(x_coord[2], y_coord[2], color=self.line_color, zorder=zorder_line)
        
        ## plot left side goal post
        ax.plot(x_coord[3], y_coord[3], color=self.line_color, zorder=zorder_line)
        
        ## right hand 6 yard box
        ax.plot(x_coord[4], y_coord[4], color=self.line_color, zorder=zorder_line)
        
        ## left side 6 yard box
        ax.plot(x_coord[5], y_coord[5], color=self.line_color, zorder=zorder_line)
        
        ## plot halfway line
        ax.plot(half_x, half_y, color=self.line_color, zorder=zorder_line)
        
        ## plot penalty and kick-off spot
        ax.scatter(spot_x[0], spot_y[0], color=self.line_color, zorder=zorder_line)
        ax.scatter(spot_x[1], spot_y[1], color=self.line_color, zorder=zorder_line)
        ax.scatter(spot_x[2], spot_y[2], color=self.line_color, zorder=zorder_line)
        
        ## plot reqired circles and arcs
        circle_1 = plt.Circle((spot_x[0], spot_y[0]), 9.15, ls='solid', lw=1.5, color=self.line_color, 
                             fill=False, zorder=zorder_line-1)
        circle_2 = plt.Circle((spot_x[1], spot_y[1]), 9.15, ls='solid', lw=1.5, color=self.line_color, 
                             fill=False, zorder=zorder_line-1)
        circle_3 = plt.Circle((spot_x[2], spot_y[2]), 9.15, ls='solid', lw=1.5, color=self.line_color, 
                             fill=False, zorder=zorder_line-1)
        
        ## add circles
        ax.add_artist(circle_1)
        ax.add_artist(circle_2)
        ax.add_artist(circle_3)
        
        ## add two rectangles to make arcs
        rect_1 = plt.Rectangle((r_x[0], r_y[0]), r_o[0], r_o[1], ls='-', 
                               color=self.pitch_color, zorder=zorder_line-1)
        rect_2 = plt.Rectangle((r_x[1], r_y[1]), r_o[0], r_o[1], ls='-', 
                               color=self.pitch_color, zorder=zorder_line-1)
        ax.add_artist(rect_1)
        ax.add_artist(rect_2)
        
        # add ractangle to fill in the color
        rect_3 = plt.Rectangle((fill_rect[0], fill_rect[1]), length, bredth, 
                               color=self.pitch_color, zorder=zorder_line-2)
        ax.add_artist(rect_3)
        
        ## tidy axis
        ax.axis('off')
        
        return fig, ax

def xG_plot(df, col, title, path=None):
    """
    Function for making an xG plot.

    Args:
        df (pandas.DataFrame): required dataframe
        col (str): xG value column name.
        title (str): title of the plot.
        path (str, optional): path where file will be saved. Defaults to None.
    """    
    ## create pitchmap
    pitch_train_real = Pitch(line_color='red', figsize_x=20.8, figsize_y=13.6)
    fig, ax = pitch_train_real.create_pitch()

    ax.text(
        52, 69.5,
        title,
        color="#121212",
        fontsize=25,
        fontfamily="Liberation Serif",
        ha="center", va="center"
    )

    ## some variable for making cmap
    start = 0.0
    stop = 1.0
    number_of_lines = 1000

    ## make the required cmap
    cm_subsection = np.linspace(start, stop, number_of_lines) 
    color = [cm.jet(x) for x in cm_subsection]
    cmap = colors.ListedColormap(color)

    ## make scatterplot
    ax.scatter(x=df['x'], y=df['y'], c=df[col], s=15,
                        marker='o', cmap=cmap, ec='#000000', lw=0.3, zorder=4)

    ## add colorbar
    fraction = 0.029
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    ax.figure.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax, pad=0, fraction=fraction)

    ## name to the color bar
    ax.text(
        111, -2,
        "xG Value",
        color="#121212",
        fontsize=25,
        fontweight="bold",
        fontfamily="Liberation Serif",
        ha="center", va="center"
    )                

    ## add credits
    ax.text(
        0.2, 0.55,
        "viz created by: @slothfulwave612",
        color="#121212",
        fontsize=10,
        fontfamily="Liberation Serif",
        ha="left", va="center"
    )

    ## save figure
    if path:
        fig.savefig(path, dpi=500, bbox_inches='tight')

def plot_dataframe(df, path):
    """
    Function to make dataframe image.

    Args:
        df (pandas.DataFrame): dataframe containing goals and xG-values
        path (str, optional): path where dataframe image will be saved. Defaults to None.
    """    
    df_styled = df.style.hide_index()
    dfi.export(df_styled, path, table_conversion='chrome', fontsize=15)

def plot_target(df, path=None):
    '''
    Function for making countplot for target column.

    Arguments:
        df -- pandas dataframe.
        path -- str, path where file will be saved.
    '''
    ## create subplot
    fig, ax = plt.subplots(figsize=(6,4))

    ## make count-plot
    sns.countplot(df['target'], ax=ax)

    ## set x-ticks
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = 'No Goal'
    labels[1] = 'Goal'
    ax.set_xticklabels(labels)

    ## set title
    ax.set_title('Countplot: Goals v No-Goals')

    ## save figure
    if path:
        fig.savefig(path, dpi=500, bbox_inches='tight')