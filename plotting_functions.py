import numpy as np, matplotlib.pyplot as plt, cartopy as cp, pandas as pd, wrf, xarray as xr, time, datetime, tqdm
import matplotlib, PIL, os #For animations
import netCDF4 as nc
from IPython.display import HTML, Image, display, Video #For displaying animations
from utils import *
matplotlib.use('Agg')
plt.rcParams["text.usetex"] = True

def area_plot(data, var, t=1, height=0, directdata=None, figure=None, lower_cut=-10000, upper_cut=10000, only_andoya=False, filled_contours=True,
              cb_levels=None, cb_extend='neither', cb_pad=0.05, cb_center=None, cmap='jet', alpha=0.8, figname=None, showfig=False, units=r'g m$^{-3}$',
              return_map=False, contour_labels=False, display_height=True):
    '''
    Plot horizontal map using Lambert Conformal projection of WRF simulation over Andoya.
    Parameters:
    -----------
    data             - (ModelOutput)     instance of ModelOutput class with WRF model simulation data
    var              - (str)             name of 3 or 4 dimensional variable to plot. If directdata is used, this will be used for creating plot title
    t                - (int)             timestep at which to plot data, default is 1
    height           - (int)             model altitude level to plot at, default is 0
    directdata       - (DataArray)       use if the variable to plot is calculated manually and not accessed from netcdf dataset, default is None
    figure           - (list-like)       plt.figure instance for creating subplots. First entry fig instance, second of form 321 or map instance. 
                                         Default is None
    lower_cut        - (float)           lowest values to plot, default is -10000
    upper_cut        - (float)           highest values to plot, default is 10000
    filled_contours  - (bool)            use either filled (True) or not filled (False) contour lines, default is True
    only_andoya      - (bool)            whether to plot full model area or only Andoya specifically, default False
    cb_levels        - (list)            specify min/max range and number of bins for colorbar, default is None
    cb_pad           - (float)           spacing between colorbar and plot in fraction of total plot width/height, default is 0.05
    cb_extend        - (str)             specify to include or exclude values outside of colorbar range specified in cb_minmax 
                                         (can be 'min', 'max', 'neither' or 'both', default 'max'
    cb_center        - (float)           set the value to use as the center of the colormap, default is None
    cmap             - (str)             name of plt.colormap instance to use for plotting, default is 'jet'
    alpha            - (float)           fraction of 1 to determine opacity of colors, default is 0.8
    figname          - (str)             pass filename if figure should be saved, default is None
    showfig          - (bool)            show figure or hold if multiple figures should be plotted, default is False
    units            - (str)             display name for the units of colorbar, default is 'g/m3'
    return_map       - (bool)            if True return the map instance for plotting more to the same axis, default is False
    contour_labels   - (bool)            if True display contour labels in unfilled contours in-line, default is False
    display_height   - (bool)            whether ot not to include height in plot title, default is True
    '''

    #Area to plot for
    if only_andoya:
        min_lat = 75.5
        max_lat = 81.2
        min_lon = 6.97
        max_lon = 32.33

    else:
        min_lat, max_lat, min_lon, max_lon = data.extent
    
    #Create map instance (choose projection)
    proj = cp.crs.LambertConformal(central_latitude=(min_lat + max_lat)/2, central_longitude=(max_lon + min_lon)/2)
    PC = cp.crs.PlateCarree()
    
    # Either create new figure and axes if nothing is passed, or use the passed figure to create new subplot
    if figure is None:
        fig = plt.figure(figsize=(5, 5))
        map = fig.add_subplot(projection=proj)
    elif type(figure[1]) != int:
        fig = figure[0]
        map = figure[1]
    else:
        fig = figure[0]
        map = fig.add_subplot(figure[1], projection=proj)
    map.set_extent([min_lon, max_lon, min_lat, max_lat], crs=PC)
    map.add_feature(cp.feature.LAND)
    map.add_feature(cp.feature.COASTLINE)
    
    cmap = plt.get_cmap(cmap)
    
    
    if directdata is not None:
        data_to_plot = directdata
    else:
        data_to_plot = data.xrData[var]
    
    if len(data_to_plot.shape) == 3:
        to_plot = data_to_plot[t, :, :]
        if directdata is None:
            to_plot.where(to_plot <= lower_cut)
            to_plot.where(to_plot <= lower_cut)

    elif len(data_to_plot.shape) == 4:
        to_plot = data_to_plot[t, height, :, :]
        if directdata is None:
            to_plot.where(to_plot <= lower_cut)
            to_plot.where(to_plot <= lower_cut)
    
    if filled_contours:
        if cb_center is not None:
            norm = matplotlib.colors.TwoSlopeNorm(vcenter=cb_center)
            plot = map.contourf(data.xrData['XLONG'].isel(Time=1), data.xrData['XLAT'].isel(Time=1), to_plot, 10, transform=PC, levels=cb_levels, 
                            alpha=alpha, cmap=cmap, extend=cb_extend, norm=norm)
        else:
            plot = map.contourf(data.xrData['XLONG'].isel(Time=1), data.xrData['XLAT'].isel(Time=1), to_plot, 10, transform=PC, levels=cb_levels, 
                            alpha=alpha, cmap=cmap, extend=cb_extend)
        if directdata is None:
            if units is None:
                plt.colorbar(plot, ax=map, orientation='horizontal', label=data.xrData[var].units, fraction=0.043, pad=cb_pad)
            else:
                plt.colorbar(plot, ax=map, orientation='horizontal', label=units, fraction=0.043, pad=cb_pad)
        else:
            plt.colorbar(plot, ax=map, orientation='horizontal', label=units, fraction=0.043, pad=cb_pad)
    
    else:
        if cb_center is not None:
            norm = matplotlib.colors.TwoSlopeNorm(vcenter=cb_center)
            plot = map.contour(data.xrData['XLONG'].isel(Time=1), data.xrData['XLAT'].isel(Time=1), to_plot, 10, transform=PC, levels=cb_levels, 
                            alpha=alpha, cmap=cmap, extend=cb_extend, norm=norm)

        else:
            plot = map.contour(data.xrData['XLONG'].isel(Time=1), data.xrData['XLAT'].isel(Time=1), to_plot, 10, transform=PC, levels=cb_levels, 
                            alpha=alpha, colors='k', extend=cb_extend)
        if contour_labels:
            map.clabel(plot, plot.levels, inline=True, fontsize=5)
        
    #Find and set date and time as title
    # if display_height:
        # plt.title(var + ' ' + str(data.date + datetime.timedelta(minutes=int(data.xrData['XTIME'][t]))))# + ' H=' + str(int(wrf.getvar(data.Data, 'z')           #                [height, 10,10])) + 'm')
    # else:
    #     plt.title(var + ' ' + str(data.date + datetime.timedelta(minutes=int(data.xrData['XTIME'][t]))))

    NYA_lonlat = (11.909895, 78.923538)
    city = map.plot(*NYA_lonlat, 'ro', transform=PC)
    fig.tight_layout()
    if figname is not None:
        plt.savefig('plots/' + figname)
    if showfig:
        plt.show()
    if return_map:
        return map


def transect_plot(data, var, t=None, directdata=None, start_coord=None, end_coord=None, pivot_coord=(78.923538, 11.909895), angle=None, 
                  prognostic=False, cmap='jet', maxheight=-1, figname=None, showfig=True, filled_contours=True, cb_levels=10, cb_extend='neither',                               cb_center=None, figure=None, axs=None, units='g m$^{-3}$', lower_cut=-10000, upper_cut=10000, return_coords=False, time_legend=False,                           legend_bbox=(0, 1.04), xlabel=True, return_ax=False, show_labels=True, title=None):
    ''' 
    Apply the wrf python module functions vertcross, getvar and CoordPair.
    Computes the transect line between start_coord and end_coord (lat, lon) or through pivot point (lat, lon) and angle and plot a variable.
    Parameters:
    -----------
    data             - (ModelOutput obj) instance of ModelOutput class with WRF model simulation data
    var              - (string) name of 3 or 4 dimensional variable to plot. If directdata is used, this will be used for creating plot title
    t                - (int) timestep at which to plot data, default is 1
    directdata       - (xarray object) use if the variable to plot is calculated manually and not accessed from netcdf dataset, default is None
    start_coord      - (list-like) coordinates of start point of transect, default is None
    end_coord        - (list-like) coordinates of end point of transect, default is None
    pivot_coord      - (list-like) coordinates of center location for transect, default is (78.923538, 11.909895) (Ny-Ålesund)
    angle            - (list-like) angle in degrees to tilt transect (positive clockwise, 0 north), default is 0
    prognostic       - (Boolean) if True the variable is calculated using wrf.getvar() 
                                 (must be a prognostic variable available from wrf module), default is False 
    cmap             - (string) matplotlib color map name to use for plotting
    maxheight        - (int) maximum height in model levels to which the transect should be plotted (use -1 if all), default is -1
    figname          - (string) pass filename if figure should be saved, default is None
    showfig          - (boolean) show figure or hold if multiple figures should be plotted, default is True
    filled_contours  - (boolean) use either filled (True) or not filled (False) contour lines, default is True
    cb_levels        - (list-like) intervals to use for colorbar, default is None
    cb_extend        - (string) specify to include or exclude values outside of colorbar range specified in cb_levels 
                                (can be 'min', 'max', 'neither' or 'both', default 'neither')
    cb_center        - (float) set the value to use as the center of the colormap, default is None
    fig              - (list-like) specify figure and add subplot to it. First entry is plt.figure instance, 
                                   second is 3 digit integer to specify subplot placement, default is None.
    axs              - (plt.axes obj) specify axes instance to plot on if needed, default is None
    units            - (string) display name for the units of colorbar, default is 'g/m3'
    cutoff           - (float) lowest values to plot, default is -10000
    return_coords    - (boolean or array) return the xy_loc coordinates to use for later plotting, default False
    time_legend      - (boolean) turn on or off legend with timestamp, default is False
    legend_bbox      - (tuple) contains the bbox legend placement in axis coordinates, default is (1, 1.5)
    xlabel           - (boolean) show the xlabel or not (useful for sharex subplots), default is True
    show_labels      - (boolean) display inline unfilled contour labels or not, default is True
    '''
    # Terrain
    ter = data.xrData['HGT'].isel(Time=1) #wrf.getvar(data.Data, "ter", timeidx=-1)
    z = wrf.getvar(data.Data, 'z')
    H = z[:, 10, 10]
    
    mHidx = maxheight #len(H[H <= maxheight])


    if prognostic:
        ds = wrf.getvar(data.Data, var)[t, :mHidx]
        
    elif directdata is not None:
        ds = directdata[t, :]
        maxlevel = len(ds[:]) - 1
        if maxlevel <= mHidx or mHidx < 0:
            mHidx = maxlevel
        
    else:
        ds = data.xrData[var][t, :mHidx] #Must use xrData in order to preserve metadata for ticks 
        
    if mHidx < 0:
        mHidx = len(ds[:]) - 1
    
    if start_coord != None and end_coord != None:
        start_point = wrf.CoordPair(lat=start_coord[0], lon=start_coord[1])
        end_point = wrf.CoordPair(lat=end_coord[0], lon=end_coord[1])
        
        cross = wrf.vertcross(ds[:mHidx], z[:mHidx], wrfin=data.Data, start_point=start_point, end_point=end_point, latlon=True, meta=True)
        ter_line = wrf.interpline(ter, wrfin=data.Data, start_point=start_point, end_point=end_point)
    
    elif pivot_coord != None and angle != None:
        pivot_point = wrf.CoordPair(lat=pivot_coord[0], lon=pivot_coord[1])
        
        cross = wrf.vertcross(ds[:mHidx], z[:mHidx], wrfin=data.Data, pivot_point=pivot_point, angle=angle, latlon=True, meta=True)
        ter_line = wrf.interpline(ter, wrfin=data.Data, pivot_point=pivot_point, angle=angle, latlon=True, meta=True)
    
    if figure is not None:
        fig = figure[0]
        ax = fig.add_subplot(figure[1])
        fig.tight_layout()
    
    elif axs is None: 
        fig = plt.figure(figsize=(10,3))
        ax = plt.axes()
        fig.tight_layout()
    
    else: #axs is not None
        ax = axs

    to_plot = wrf.to_np(cross)
    to_plot[to_plot <= lower_cut] = np.nan
    to_plot[to_plot >= upper_cut] = np.nan
    
    if filled_contours:
        if cb_levels is not None:
            if cb_center is not None:
                norm = matplotlib.colors.TwoSlopeNorm(vcenter=cb_center)
                contour = ax.contourf(to_plot[:mHidx], cmap=plt.get_cmap(cmap), levels=cb_levels, extend=cb_extend, norm=norm)
            else:
                contour = ax.contourf(to_plot[:mHidx], cmap=plt.get_cmap(cmap), levels=cb_levels, extend=cb_extend)
        else:
            contour = ax.contourf(to_plot[:mHidx], cmap=plt.get_cmap(cmap))
            
        if directdata is not None:
            plt.colorbar(contour, ax=ax, label=units)
        else:
            plt.colorbar(contour, ax=ax, label=data.xrData[var].units)
    else:
        if cb_levels is not None:
            if cb_center is not None:
                norm = matplotlib.colors.TwoSlopeNorm(vcenter=cb_center)
                contour = ax.contour(to_plot[:mHidx], colors='k', levels=cb_levels, extend=cb_extend, norm=norm)
            else:
                contour = ax.contour(to_plot[:mHidx], colors='k', levels=cb_levels, extend=cb_extend)
        else:
            contour = ax.contour(to_plot[:mHidx], colors='k')
        if show_labels:
            ax.clabel(contour, contour.levels, inline=True, fontsize=5)
 
        
    #Terrain
    ht_fill = ax.fill_between(np.arange(0, cross.shape[1], 1), 0, ter_line / 28, facecolor="saddlebrown")
        
    #---------------------------------------------------------------------
    # Set the x-ticks to use latitude and longitude labels.
    if type(return_coords) == tuple:
        # Workaround to avoid missing metadata in IWF calculation.
        coord_pairs = return_coords[0]
    else:
        coord_pairs = wrf.to_np(cross.coords["xy_loc"])
        
    x_ticks = np.arange(coord_pairs.shape[0])
    x_labels = [pair.latlon_str(fmt="{:.2f}N, {:.2f}E") for pair in wrf.to_np(coord_pairs)]
    ax.set_xticks(x_ticks[::20])
    ax.set_xticklabels(x_labels[::20], rotation=0, fontsize=8)

    # Set the y-ticks to be height.
    if type(return_coords) == tuple:
        vert_vals = return_coords[1]
    else:
        vert_vals = wrf.to_np(cross[:mHidx,:].coords["vertical"]).astype(int)
    v_ticks = np.arange(vert_vals.shape[0])
    ax.set_yticks(v_ticks[::int(mHidx/5)])
    ax.set_yticklabels(vert_vals[::int(mHidx/5)], fontsize=8)
    #------------------------------------------------------------------------
    if title is None:
        ax.set_title(var)
    else:
        ax.set_title(title)
    ax.set_ylabel("Height (m)", fontsize=12)
    if xlabel:
        ax.set_xlabel("Latitude, Longitude", fontsize=12)
    if time_legend:
        ax.text(*legend_bbox, data.date + datetime.timedelta(minutes=t*5), transform=ax.transAxes)
    if axs is None:
        fig.tight_layout()
    if showfig:
        plt.show()
    if figname is not None:
        plt.savefig(f'plots/{figname}.png')
    if return_coords:
        return coord_pairs, vert_vals
    
    
def time_cross(data, var, directdata=None, loc=(78.923538, 11.909895), cutoff=0.001, maxheight=1000, figure=None, ax=None, 
               alpha=1, xlabel=None, showfig=False, units='g m$^{-3}$', filled_contours=True, cmap='jet', cb_levels=11, 
               cb_extend='neither', cb_orientation='vertical', cb_location='right', cb_pad=0.01, cb_center=None, plot_pblh=False, 
               skip_spinup=0, title=None):
    '''
    Create an animation of time evolution of variable across the vertical transect.
    Parameters:
    -----------
    data             - (ModelOutput obj) instance of ModelOutput class with WRF model simulation data
    var              - (string) name of 3 or 4 dimensional variable to plot. If directdata is used, this will be used for creating plot title
    directdata       - (xarray object) use if the variable to plot is calculated manually and not accessed from netcdf dataset, 
                                       default is None
    loc              - (list-like) lat, lon coordinates in decimal degrees specifying location for plot, 
                                   default is (78.923538, 11.909895) (Ny-Ålesund)
    cutoff           - (float) lowest values to plot, default is 0.001
    maxheight        - (int) maximum height in meters to which the transect should be plotted, default is 1000
    figure           - (plt.figure obj) figure to generate plots for, default is None
    ax               - (plt.axes obj) axes to generate plots on. Must be specified if figure is not None, default is None
    alpha            - (float) specify opacity of contours. 0 is fully transparent, 1 is fully opaque, default is 1
    xlabel           - (string) name of label for x-axis, default is None
    showfig          - (boolean) if True, call plt.show() to display the plot, default is False
    units            - (string) display name for the units of colorbar, default is 'g/m3'
    filled_contours  - (boolean) use either filled (True) or not filled (False) contour lines, default is True
    cmap             - (string) name of plt.cmap colormap, default is 'jet'
    cb_levels        - (list-like) intervals for contour color bins, default is 11
    cb_extend        - (string) specify to include or exclude values outside of colorbar range specified in cb_minmax 
                                (can be 'min', 'max', 'neither' or 'both', default 'max'
    cb_orientation   - (string) specify orientation of colorbar, either 'vertical' or 'horizontal', default is 'vertical'
    cb_location      - (string) specify placement of colorbar, either 'bottom', 'top', 'right', or 'left', default is 'right'
    cb_pad           - (float) specify padding space between plot and colorbar in fraction of total plot width/height, default is 0.01
    cb_center        - (float) set the value to use as the center of the colormap, default is None
    plot_pblh        - (boolean) plot the planetary boundary layer height along with the other variable, default is False
    skip_spinup      - (int) number of timesteps to skip due to model spinup, default is 0
    '''
    x, y = wrf.ll_to_xy(data.Data, *loc)
    H = wrf.getvar(data.Data, 'z')[:, y, x]
    mHidx = len(H[H <= maxheight])
    
    if directdata is not None:
        to_plot = wrf.to_np(directdata[skip_spinup:, :, x, y])
        to_plot[to_plot <= cutoff] = np.nan
    else:
        to_plot = data.xrData[var].isel(Time=slice(skip_spinup, None), west_east=x, south_north=y)
        
        #to_plot = to_plot.where(to_plot <= cutoff, to_plot, np.nan)
        
    TIME, HEIGHT = np.meshgrid(data.xrData['XTIME'][skip_spinup:], H[:mHidx])
    
    if figure is None:
        fig, ax = plt.subplots(figsize=(10,2))
    else:
        fig = figure
    
    if filled_contours:
        if cb_center is not None:
            norm = matplotlib.colors.TwoSlopeNorm(vcenter=cb_center)
            plot = ax.contourf(TIME.T, HEIGHT.T, to_plot[:, :mHidx], cmap=plt.get_cmap(cmap), levels=cb_levels, extend=cb_extend, alpha=alpha, norm=norm)
        else:
            plot = ax.contourf(TIME.T, HEIGHT.T, to_plot[:, :mHidx], cmap=plt.get_cmap(cmap), levels=cb_levels, extend=cb_extend, alpha=alpha)
    else:
        plot = ax.contour(TIME.T, HEIGHT.T, to_plot[:, :mHidx], levels=cb_levels, colors='k', extend=cb_extend, alpha=alpha, linewidths=0.5)
        ax.clabel(plot, plot.levels, inline=True, fontsize=7)
    
    if plot_pblh:
        ax.plot(data.xrData['XTIME'][skip_spinup:], data.xrData['PBLH'][skip_spinup:, x, y], 'brown', linestyle='--', linewidth=1)
        
#     x_ticks = (np.arange(to_plot.shape[0]) + skip_spinup) * 5 # Convert timestep number to minutes
#     x_labels = [(Mil3.date + datetime.timedelta(minutes=int(t))).strftime('%H:%M') for t in Mil3.xrData['XTIME'][skip_spinup:]]
#     ax.set_xticks(x_ticks[::24])
#     ax.set_xticklabels(x_labels[::24], rotation=25, fontsize=8)
    
    ax.set_ylabel('Altitude [m]')
    #ax.set_xlabel('')
        
    fig.tight_layout()
    if title is None:
        ax.set_title(var)
    else:
        ax.set_title(title)
    if directdata is not None:
        if filled_contours:
            cbar = plt.colorbar(plot, orientation=cb_orientation, label=units, ax=ax, location=cb_location, pad=cb_pad)
    else:
        if units is None:
            units = data.xrData[var].units
        if filled_contours:
            cbar = plt.colorbar(plot, orientation=cb_orientation, label=units, ax=ax, location=cb_location, pad=cb_pad)
            
    if showfig:
        plt.show()