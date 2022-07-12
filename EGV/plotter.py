import pandas as pd
import matplotlib
import plotly
import tkinter
import os
import numpy as np
import seaborn
import plotly.express
import kaleido
import nbformat
import telecontrol_parser as tp


def main():
    save_TS_plots = False
    save_violin_plots = False
    save_quantile_plots = False
    locatie = 'parkhaven'
    egv_db = tp.egv_standard_run(locatie)
    my_dpi = initialize_params(locatie)
    numeric_cols = egv_db.select_dtypes('number').columns

    subsets = (['2015-01-01', '2015-12-31'], ['2016-01-01', '2016-12-31'],
               ['2017-01-01', '2017-12-31'], ['2018-01-01', '2018-12-31'],
               ['2019-01-01', '2019-12-31'], ['2020-01-01', '2020-12-31'],
               ['2021-01-01', '2021-12-31'], ['2022-01-01', '2022-12-31'],
               ['2015-01-01', '2022-12-31']
               )
    plot_TS(locatie, egv_db, numeric_cols, subsets)


def initialize_params(locatie):
    with open('teams_path') as f:
        lines = f.readlines()
    teams_path = lines[0] + '/'
    isExist = os.path.exists(teams_path+'plots/' + locatie)
    if not isExist:
        os.makedirs(teams_path+'plots/' + locatie)
    pd.options.plotting.backend = "matplotlib"
    root = tkinter.Tk()
    my_dpi = root.winfo_fpixels('1i')
    return (my_dpi, teams_path)


def plot_TS(locatie, egv_db, numeric_cols, subsets=None):
    my_dpi, teams_path = initialize_params(locatie)
    if isinstance(subsets, type(None)):
        subsets = [[str(egv_db.index[0])[0:10], str(egv_db.index[-1])[0:10]]]
    for subset in subsets:
        plot1 = egv_db[subset[0]:subset[1]].plot(
            'datetime', y=numeric_cols[1],
            figsize=(1920/my_dpi, 1080/my_dpi))
        axes = plot1.xaxis
        axes.set_major_locator(matplotlib.dates.MonthLocator(interval=1))
        axes.get_major_ticks()[-1].label1.set_visible(False)
        plot1.set_xlabel('Datum')
        plot1.set_ylabel('EGV')
        plot1.set_title(
            'Van ' + subset[0] + ' t/m ' + subset[1] +
            '\n Locatie: ' + locatie)
        plot = plot1.figure
        plot_int = plotly.tools.mpl_to_plotly(plot)
        plot_int.write_html(teams_path+"plots/" + locatie +
                            '/' + locatie + subset[0] +
                            '_' + subset[1] + ".html")
        plot_int.write_image(teams_path+"plots/" +
                             locatie + '/' + locatie +
                             subset[0] + '_' + subset[1] + ".png")


if __name__ == '__main__':
    main()
