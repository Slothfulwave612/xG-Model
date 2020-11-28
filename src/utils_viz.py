'''
__author__: slothfulwave612
'''

## required packages/modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
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
    
    def __init__(self, line_color="#000000", pitch_color="#FFFFFF", orientation="horizontal", half=False, plot_arrow=True):
        '''
        Function to initialize the class object.
        
        Arguments:
            self -- represents the object of the class.
            line_color -- str, color of the lines. Default: '#000000'
            pitch_color -- str, color of the pitch-map. Default: '#FFFFFF'
            orientation -- str, pitch orientation 'horizontal' or 'vertical'. Default: 'horizontal'
            half 
        '''
        
        self.line_color = line_color
        self.pitch_color = pitch_color
        self.orientation = orientation
        self.half = half
        self.plot_arrow = plot_arrow
        
        ## figsize
        self.figsize_x = 25             ## x-axis length
        self.figsize_y = 12             ## y-axis length
        
        ## pitch outfield lines
        self.pitch_dims = [
            [0, 104, 104, 0, 0],           ## x-axis
            [0, 0, 68, 68, 0]              ## y-axis
        ]
        
        self.dims_x = [
            [104, 87.5, 87.5, 104],        ## right-side penalty box(x-axis)
            [0, 16.5, 16.5, 0],            ## left-side penalty box(x-axis)
            [104, 104.2, 104.2, 104],      ## right-side goal post(x-axis)
            [0, -0.2, -0.2, 0],            ## left-side goal post(x-axis)
            [104, 99.5, 99.5, 104],        ## right-side 6-yard-box(x-axis)
            [0, 4.5, 4.5, 0]               ## left-side 6-yard-box(x-axis)
        ]
        
        self.dims_y = [
            [13.84, 13.84, 54.16, 54.16],  ## right-side penalty box(y-axis)
            [13.84, 13.84, 54.16, 54.16],  ## left-side penalty box(y-axis)
            [30.34, 30.34, 37.66, 37.66],  ## right-side goal post(y-axis)
            [30.34, 30.34, 37.66, 37.66],  ## left-side goal post(y-axis)
            [24.84, 24.84, 43.16, 43.16],  ## right-side 6-yard-box(y-axis)
            [24.84, 24.84, 43.16, 43.16]   ## left-side 6-yard-box(y-axis)
        ]
        
        self.half_x = [52, 52]             ## halfway line x-axis
        self.half_y = [0, 68]              ## halfway line y-axis
        
        self.scatter_x = [93, 10.5, 52]    ## penalty and kick off spot(x-axis)
        self.scatter_y = [34, 34, 34]      ## penalty and kick off spot(y-axis)
        
    def create_pitch(self, zorder_line=3, figax=None):
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
            
            
        elif self.orientation == 'vertical':
            ## figsize
            sx, sy = self.figsize_y, self.figsize_x
            
            ## x and y coordinates for pitch outline
            if self.half == False:
                pitch_dims_x = self.pitch_dims[1]
                pitch_dims_y = self.pitch_dims[0]
            else:
                pitch_dims_x = [0, 0, 68, 68]
                pitch_dims_y = [52, 104, 104, 52]
            
            ## x and y coordinates for halfway line
            half_x = self.half_y
            half_y = self.half_x
            
            ## x and y coordinate list
            x_coord = self.dims_y
            y_coord = self.dims_x
        
            ## x and y coordinate for spots
            spot_x = self.scatter_y
            spot_y = self.scatter_x 
        
        else:
            raise Exception('Orientation not understood!!!')
        
        if figax == None:
            ## create subplot
            fig, ax = plt.subplots(figsize=(sx, sy), facecolor=self.pitch_color)
            ax.set_facecolor(self.pitch_color)
            ax.set_aspect("equal")
        else:
            fig, ax = figax[0], figax[1]

        ## plot outfield lines
        ax.plot(pitch_dims_x, pitch_dims_y, color=self.line_color, zorder=zorder_line)
        
        ## plot right side penalty box
        ax.plot(x_coord[0], y_coord[0], color=self.line_color, zorder=zorder_line)            
        
        ## plot right side goal post
        ax.plot(x_coord[2], y_coord[2], color=self.line_color, zorder=zorder_line)                
        
        ## right hand 6 yard box
        ax.plot(x_coord[4], y_coord[4], color=self.line_color, zorder=zorder_line)                
        
        ## plot halfway line
        ax.plot(half_x, half_y, color=self.line_color, zorder=zorder_line)

        if self.half == False:
            ## plot left side penalty box
            ax.plot(x_coord[1], y_coord[1], color=self.line_color, zorder=zorder_line)

            ## plot left side goal post
            ax.plot(x_coord[3], y_coord[3], color=self.line_color, zorder=zorder_line)

            ## left side 6 yard box
            ax.plot(x_coord[5], y_coord[5], color=self.line_color, zorder=zorder_line)

            ## kick-off spot
            ax.scatter(spot_x[1], spot_y[1], color=self.line_color, s=5, zorder=zorder_line)
        
        ## plot penalty and kick-off spot
        ax.scatter(spot_x[0], spot_y[0], color=self.line_color, s=5, zorder=zorder_line)
        ax.scatter(spot_x[2], spot_y[2], color=self.line_color, s=5, zorder=zorder_line)
        
        if self.half == False:
            ## plot center circle
            circle = plt.Circle((spot_x[2], spot_y[2]), 9.15, lw=1.5, color=self.line_color, 
                                fill=False, zorder=zorder_line)
            ax.add_patch(circle)
        else:
            ## draw center arc
            arc = Arc(xy=(34, 52), height=18.5, width=18.5, angle=90, theta1=270, theta2=90, color=self.line_color, zorder=zorder_line)
            ax.add_patch(arc)

        ## adding arcs
        if self.orientation == "horizontal":
            arc_left = Arc(xy=(10.5, 34), height=18.5, width=18.5, angle=0, theta1=310, theta2=50, color=self.line_color, zorder=zorder_line)
            arc_right = Arc(xy=(93.5, 34), height=18.5, width=18.5, angle=0, theta1=130, theta2=230, color=self.line_color, zorder=zorder_line)

            ax.add_patch(arc_left)
            ax.add_patch(arc_right)
        
        elif self.orientation == "vertical":
            if self.half == False:
                arc_bottom = Arc(xy=(34, 10.5), height=18.5, width=18.5, angle=90, theta1=310, theta2=50, color=self.line_color, zorder=zorder_line)
                ax.add_patch(arc_bottom)

            arc_top = Arc(xy=(34, 93.5), height=18.5, width=18.5, angle=90, theta1=130, theta2=230, color=self.line_color, zorder=zorder_line)
            ax.add_patch(arc_top)

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
    pitch_train_real = Pitch(line_color='red')
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
    fraction = 0.02
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    ax.figure.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax, pad=0, fraction=fraction)

    ## name to the color bar
    ax.text(
        111, -1,
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


def plot_correlation(df, numerical_feature_columns, target, filename=None):
    """[summary]

    Args:
        df ([type]): [description]
        numerical_feature_columns ([type]): [description]
        target ([type]): [description]
        filename ([type], optional): [description]. Defaults to None.
    """    
    cm = df[numerical_feature_columns].corr()
    fig, ax = plt.subplots(figsize=(16,12))
    sns.heatmap(cm, annot=True, cmap = 'viridis')
    
    if filename:
        fig.savefig(filename, dpi=500, bbox_inches="tight")
    
    return fig, ax

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