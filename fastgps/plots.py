import torch 

def plot_fastgps_fit_data(
        fit_data, 
        keys, 
        fsf_cols = 5, 
        fsf_rows = 5, 
        tight_layout = True, 
        savepath = None,
        dict_log_scale = None, 
        fontsize = "xx-large",
        linewidth = 3,
        ):
    """ 
    Plot loss and hyperparameter optimization 
    
    Arg:
        fit_data (dict): data returned by calling the `fit` method on a GP.
        keys (list): list of keys to plot.
        fsf_cols (float): `figsize=(fsf_cols*ncols,fsf_rows)`.
        fsf_rows (float): `figsize=(fsf_cols*ncols,fsf_rows)`.
        savepath (str): path where the plot is saved; if `None`, then the plot is not saved.  
        dict_log_scale (dict): dictionary where `dict_log_scale[key]` is a `bool` indicating 
            whether or not to log scale the axis corresponding to this key. 
            If `None`, the it will be log scaled whenever the data is all positive
        fontsize (str): font size 
        linewidth (float): line width
    
    Returns:
        fig (matplotlib.figure.Figure): figure 
        ax (np.ndarray): array of `matplotlib.axes._axes.Axes`
    """
    from matplotlib import pyplot
    if isinstance(keys,str):
        keys = [keys] 
    keys = list(keys)
    assert isinstance(keys,list) and all(isinstance(key,str) for key in keys)
    for key in keys: 
        if key not in fit_data:
            raise Exception("key = %s not in fit_data.keys() = %s"%(key,str(list(fit_data.keys()))))
    ncols = len(keys) 
    nrows = 1 
    fig,ax = pyplot.subplots(nrows=nrows,ncols=ncols,figsize=(fsf_cols*ncols,fsf_rows)) 
    for i,key in enumerate(keys):
        data = fit_data[key]
        if data.ndim>1: data =data.flatten(start_dim=1)
        data = data.detach().to("cpu")
        ax[i].plot(fit_data["iteration"],data,linewidth=linewidth)
        if (dict_log_scale is None and (data>0).all()) or (dict_log_scale is not None and dict_log_scale[key]==True):
            ax[i].set_yscale("log",base=10)
        ax[i].set_xlabel("iterations",fontsize=fontsize)
        ax[i].set_ylabel(key.replace("_"," ").replace(" hist",""),fontsize=fontsize)
    if tight_layout:
        fig.tight_layout()
    if savepath is not None: 
        fig.savefig(savepath,bbox_inches="tight")
    return fig,ax
