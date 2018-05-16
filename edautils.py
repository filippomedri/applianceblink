
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def set_plot():
    sns.set(color_codes=True)
    plt.figure(figsize=(30, 22))

def plot_measure_hist(df,measure,bins=17):
    plt.suptitle(measure + ' Distribution')
    plt.xlabel('Watt')
    df[measure].hist(bins=bins)


def plot_daily_values(df,datetime,measure,threshold,top, month, day):

    df_monthly = df[(df[datetime].dt.month == month)]
    df_daily = df_monthly[(df_monthly[datetime].dt.day == day)]

    values = df_daily[measure]
    x = df_daily[datetime].dt.hour

    # split it up
    above_threshold = np.maximum(values - threshold, 0)
    below_threshold = np.minimum(values, threshold)

    # and plot it
    fig, ax = plt.subplots()
    plt.suptitle(measure + '\n' + str(month) + '-' +str(day) + '-'+ '2016')
    plt.ylabel('Peak Power Consumption\n (Watt)')
    plt.xlabel('Hour')
    ax.set_xlim([0, 24])
    ax.set_ylim([5, top + 1])
    ax.bar(x, below_threshold, 0.35, color="g")
    ax.bar(x, above_threshold, 0.35, color="r",
           bottom=below_threshold)

    # horizontal line indicating the threshold
    ax.plot([0., 24.], [threshold, threshold], "k--")

def plot_hour_series(df,datetime,measure,threshold, top, month, day, start_hour, end_hour):

    df_monthly = df[(df[datetime].dt.month == month)]
    df_daily = df_monthly[(df_monthly[datetime].dt.day == day)]
    hours = range(start_hour,end_hour+1)
    df_hour = df_daily[df_daily[datetime].dt.hour.isin(hours)]

    values = df_hour[measure]
    x = [m for m in range(60*len(hours))]

    #x = df_hour[datetime].dt.minute

    # split it up
    above_threshold = np.maximum(values - threshold, 0)
    below_threshold = np.minimum(values, threshold)

    # and plot it
    fig, ax = plt.subplots()
    plt.suptitle(measure + '\n' + str(month) + '-' + str(day) + '-' + '2016'
                 + '\n' + 'hours: ' + str(start_hour) + '-' + str(end_hour))
    plt.ylabel('Watt')
    plt.xlabel('Minute')
    ax.set_xlim([0, 60*len(hours)-1])
    ax.set_ylim([5, top+1])
    ax.bar(x, below_threshold, 0.35, color="g")
    ax.bar(x, above_threshold, 0.35, color="r",
           bottom=below_threshold)

    # horizontal line indicating the threshold
    ax.plot([0, 60*len(hours)-1], [threshold, threshold], "k--")

def compare_measure_and_appliance_activation(df,datetime, measure, appliance, appliance_threshold):
    df.loc[df[appliance] < appliance_threshold, appliance + ' State'] = "Down"
    df.loc[df[appliance] >= appliance_threshold, appliance + ' State'] = "Up"

    df['Hour'] = df[datetime].dt.hour
    sns.lmplot(x="Hour", y=measure,  data=df, fit_reg=False, hue= appliance +' State',
               palette=dict(Down="g", Up="r"),markers=["o","x"],hue_order=['Down','Up']
    )
    ax = plt.gca()
    ax.set_title(appliance + ' Activation')