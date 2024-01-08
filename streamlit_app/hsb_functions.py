import streamlit as st
import numpy as np
from scipy.stats import shapiro, mannwhitneyu, chi2_contingency, spearmanr
import pandas as pd

def pval_txt(pval):
    """
    Formats a p-value according to its value.

    If the p-value is less than 0.0001, the function returns the string 'p < 10^-4'.
    Otherwise, it returns the p-value rounded to 4 decimal places as a string, prefixed with 'p = '.

    Args:
        pval (float): The p-value to format.

    Returns:
        str: The formatted p-value.
    """
    if pval < 0.0001:
        return f'<i>p</i> < 10\u207b\u2074'
    else:
        return f'<i>p</i> = {pval:.4f}'
    
def pval_shapiro(df, var):
    pval = shapiro(df[var.name])[1]
    return f"Normally distributed <br>{pval_txt(pval)}" if pval >= 0.05 else f"Not normally distributed <br>{pval_txt(pval)}"

def mean_sd_range(df, var):
    return (f"{np.mean(df[var.name]):.1f} ± {np.std(df[var.name]):.1f}", f"[{np.min(df[var.name]):.1f} - {np.max(df[var.name]):.1f}]")

def univar_cont(df, var_lst):
    body = f""""""
    head = f"""
    <table class = "table_1" style = "width: 75% !important">
        <tr class = "head_tr">
            <th>Parameter</th>
            <th>Mean ± SD</th>
            <th>Range</th>
            <th>Normality</th>
        </tr>
    """    
    for var in var_lst:
        body += f"""
        <tr>
            <td>{var.full_name} ({var.unit})</td>
            <td style = 'border-left: 1px dashed black'>{mean_sd_range(df, var)[0]}</td>
            <td style = 'border-left: 1px dashed black'>{mean_sd_range(df, var)[1]}</td>
            <td style = 'border-left: 1px dashed black'>{pval_shapiro(df, var)}</td>
        </tr>
        """       
    tail = f"""</table>"""   
    table = head + body + tail  
    return table

def univar_cat(df, var_lst):
    body = f""""""
    head = f"""
    <table class = "table_1" style = "width: 75% !important">
        <tr class = "head_tr">
            <th>Parameter</th>
            <th>Modalities</th>
            <th>Observations (<i>n</i>)</th>
            <th>Observations (%)</th>
        </tr>
    """
    
    for var in var_lst:
        body += f"""
            <tr style = 'border-top: 1px solid black'>
                <td rowspan = "{len(var.mod)}">{var.full_name}</td>"""
        for mod, name in zip(var.mod, var.mod_names):
            body += f"""
                <td style = 'border-left: 1px dashed black'>{name}</td>
                <td style = 'border-left: 1px dashed black'>{df[var.name].value_counts()[mod]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var.name].value_counts(normalize = True)[mod]:.1%}</td>
            </tr>
            """
        
    tail = f"""</table>"""
    
    table = head + body + tail
    
    st.write(table, unsafe_allow_html=True)    
    
    
def mwu_cardio(df, var):
    """
    Computes a Mann-Whitney test for the difference in the input characteristic between patients with and without cardiovascular disease.

    Args:
    var: str
    The name of the characteristic to compare between patients with and without cardiovascular disease.

    Returns:
    pval: float
    The p-value of the Mann-Whitney U test.
    """
    mwu, pval = mannwhitneyu(df[df['cardio'] == "0"][var.name], df[df['cardio'] == "1"][var.name])
    return pval

def mwu(df1, df2, var):

    mwu, pval = mannwhitneyu(df1[var.name], df2[var.name])
    return pval_txt(pval)

def pp_3(df, var):
    return f"""N: {df[var.name].value_counts(normalize = True)["1"]:.1%}
    <br>+: {df[var.name].value_counts(normalize = True)["2"]:.1%}
    <br>++: {df[var.name].value_counts(normalize = True)["3"]:.1%}
    """

def pp_3_cardio(df, var, card):
    if card == "0":
        df = df[df['cardio'] == "0"]
    else:
        df = df[df['cardio'] == "1"]
    return f"""N: {df[var.name].value_counts(normalize = True)["1"]:.1%}
    <br>+: {df[var.name].value_counts(normalize = True)["2"]:.1%}
    <br>++: {df[var.name].value_counts(normalize = True)["3"]:.1%}
    """

def pp_4_cardio(df, var, card):
    if card == "0":
        df = df[df['cardio'] == "0"]
    else:
        df = df[df['cardio'] == "1"]
    return f"""N: {df[var.name].value_counts(normalize = True)["1"]:.1%}
    <br>Elevated: {df[var.name].value_counts(normalize = True)["2"]:.1%}
    <br>HTN I: {df[var.name].value_counts(normalize = True)["3"]:.1%}
    <br>HTN II: {df[var.name].value_counts(normalize = True)["4"]:.1%}
    """

def pp_4(df, var):
    return f"""N: {df[var.name].value_counts(normalize = True)["1"]:.1%}
    <br>Elevated: {df[var.name].value_counts(normalize = True)["2"]:.1%}
    <br>HTN I: {df[var.name].value_counts(normalize = True)["3"]:.1%}
    <br>HTN II: {df[var.name].value_counts(normalize = True)["4"]:.1%}
    """

def chi2_cardio(df, var):
    """
    Calculates the chi-squared statistic and p-value for a categorical variable in relation to 'cardio'.

    Parameters:
    - var (str): The name of the categorical variable in the DataFrame 'df'.

    Returns:
    - str: A text description of the p-value result.
    """
    crosstab = pd.crosstab(df[var.name], df['cardio'])
    chi2, pval, _, _ = chi2_contingency(crosstab)
    return pval_txt(pval)

def chi2_var(df, var1, var2):
    """
    Calculates the chi-squared statistic and p-value for a contingency table of two categorical variables.

    Parameters:
    - var1 (str): The name of the first categorical variable in the DataFrame 'df'.
    - var2 (str): The name of the second categorical variable in the DataFrame 'df'.

    Returns:
    - str: A text description of the p-value result.
    """
    crosstab = pd.crosstab(df[var1.name], df[var2.name])
    chi2, pval, _, _ = chi2_contingency(crosstab)
    return pval_txt(pval)

def mean_sd(df, var):
    """
    Calculates the mean and standard deviation for a numerical variable in the DataFrame 'df' 
    and return the results as a formatted string.

    Parameters:
    - var (str): The name of the numerical variable in the DataFrame 'df'.

    Returns:
    - str: A formatted string representing the mean and standard deviation in the format 'mean ± sd'.
    """
    return f'{np.mean(df[var.name]):.1f} ± {np.std(df[var.name]):.1f}'

def mean_sd_1(df, var):
    """
    Calculates the mean and standard deviation for a numerical variable in the subset of the DataFrame 'df' 
    where 'cardio' equals 1 and return the results as a formatted string.

    Parameters:
    - var (str): The name of the numerical variable in the DataFrame 'df'.

    Returns:
    - str: A formatted string representing the mean and standard deviation in the format 'mean ± sd'.
    """
    return f'{np.mean(df[df.cardio == "1"][var.name]):.1f} ± {np.std(df[df.cardio == "1"][var.name]):.1f}'

def mean_sd_0(df, var):
    """
    Calculates the mean and standard deviation for a numerical variable in the subset of the DataFrame 'df' 
    where 'cardio' equals 0 and return the results as a formatted string.

    Parameters:
    - var (str): The name of the numerical variable in the DataFrame 'df'.

    Returns:
    - str: A formatted string representing the mean and standard deviation in the format 'mean ± sd'.
    """
    return f'{np.mean(df[df.cardio == "0"][var.name]):.1f} ± {np.std(df[df.cardio == "0"][var.name]):.1f}'

def pp(df, var):
    """
    Calculates the percentage of positive occurrences (1) for a binary categorical variable in the DataFrame 'df' 
    and return the result as a formatted string.

    Parameters:
    - var (str): The name of the binary categorical variable in the DataFrame 'df'.

    Returns:
    - str: A formatted string representing the percentage of positive occurrences in the format 'xx.x%'.
    """
    return f'{df[var.name].value_counts(normalize = True)[1]:.1%}'
    
def pp_0(df, var):
    """
    Calculates the percentage of positive occurrences (1) for a binary categorical variable in the subset of 
    the DataFrame 'df' where 'cardio' equals 0, and return the result as a formatted string.

    Parameters:
    - var (str): The name of the binary categorical variable in the DataFrame 'df'.

    Returns:
    - str: A formatted string representing the percentage of positive occurrences in the format 'xx.x%'.
    """
    return f'{df[df.cardio == "0"][var.name].value_counts(normalize = True)[1]:.1%}'
    
def pp_1(df, var):
    """
    Calculates the percentage of positive occurrences (1) for a binary categorical variable in the subset of 
    the DataFrame 'df' where 'cardio' equals 1, and return the result as a formatted string.

    Parameters:
    - var (str): The name of the binary categorical variable in the DataFrame 'df'.

    Returns:
    - str: A formatted string representing the percentage of positive occurrences in the format 'xx.x%'.
    """
    return f'{df[df.cardio == "1"][var.name].value_counts(normalize = True)[1]:.1%}'

def min_max(df, var):
    return f'[{np.min(df[var.name]):.1f} - {np.max(df[var.name]):.1f}]'

def matrix_display(matrix):
    """
    Displays a confusion matrix in a formatted HTML style.

    Parameters:
    - matrix (list of lists): A 2x2 confusion matrix containing True Negatives, False Positives,
      False Negatives, and True Positives.

    Returns:
    - None
    """
    return    f""" <table style = 'border 1px solid black; width : 70%; border: 1px solid black'>
        <tr style = 'background-color: gray; color: white; border: 1px solid black'>
            <th>Classification</th>
            <th>Samples (<i>n</i>)</th>
            <th>Samples (%)</th>
        </tr>
        <tr>
            <th colspan = "3" style = 'border: 1px solid black; background-color: lightcyan; font-variant: small-caps; font-weight:bold; text-align: left ! important; line-height: 0.6'>Class 0: Absence of CV disease</th>
        </tr>
        <tr>
            <td>True Negatives</td>
            <td>{matrix[0][0]}</td>
            <td style = 'color: green; font-weight: bold'>{matrix[0][0]/matrix[0].sum():.1%}</td>
        </tr>
        <tr>
            <td>False Positives</td>
            <td>{matrix[0][1]}</td>
            <td style = 'color: firebrick; font-weight: bold'>{matrix[0][1]/matrix[0].sum():.1%}</td>
        </tr>
        <tr>
            <th colspan = "3" style = 'border: 1px solid black; background-color: lightcyan; font-variant: small-caps; font-weight:bold; text-align: left ! important; line-height: 0.6'>Class 1: Presence of CV disease</th>
        </tr>
        <tr>
            <td>True Positives</td>
            <td>{matrix[1][1]}</td>
            <td style = 'color: green; font-weight: bold'>{matrix[1][1]/matrix[1].sum():.1%}</td>
        </tr>
        <tr>
            <td>False Negatives</td>
            <td>{matrix[1][0]}</td>
            <td style = 'color: firebrick; font-weight: bold'>{matrix[1][0]/matrix[1].sum():.1%}</td>
        </tr>
    </table>"""
    
def classification_plot(matrix):

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 6))

    plt.subplots_adjust(wspace = 1)

    ax[0].pie(matrix[0], labels = ['True Negatives', 'False Positives'], autopct = '%.0f%%', startangle=90, colors = ['palegreen', 'lightcoral'], wedgeprops = {'edgecolor' : 'black'}, textprops = {'fontsize' : '14'})
    ax[0].set_title('Classification: Class 0', fontdict = fontdict_title)
    ax[1].pie(matrix[1][::-1], labels = ['True Positives','False Negatives'], autopct = '%.0f%%', startangle=90, colors = ['palegreen', 'lightcoral'], wedgeprops = {'edgecolor' : 'black'}, textprops = {'fontsize' : '14'})
    ax[1].set_title('Classification: Class 1', fontdict = fontdict_title)

    plt.show()  

def perf_barplot(classification_rep, df_cv):
    
    dfrep = pd.DataFrame(classification_rep).reset_index().rename(columns={'index': 'Metric'})
    dfrep_m = pd.melt(dfrep[['Metric', '0.0', '1.0']], id_vars = "Metric", var_name = "Class", value_name = "Score")
    dfrep_m2 = pd.merge(dfrep_m, dfrep[['Metric', 'accuracy', 'macro avg', 'weighted avg']], on = "Metric", how = "left")
    dfrep_m2.loc[dfrep_m2["Metric"] != "support", ["Score", "macro avg", "weighted avg"]] *= 100
    dfrep_m2["accuracy"] *= 100
    
    df_cv_melted = df_cv.melt(var_name='Metric', value_name='Value')
    
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,5))
    plt.subplots_adjust(wspace = 0.3)

    sns.barplot(x = 'Metric', 
                y = "Score", 
                data = dfrep_m2.loc[dfrep_m2["Metric"] != "support"], 
                hue = "Class",
                palette = ['yellow', 'darkorchid'],
                edgecolor = "black",
                ax = ax[0],
               )
    
    sns.barplot(x = 'Metric',
                y = 'Value',
                data = df_cv_melted.loc[df_cv_melted['Metric'] != "AUC"],
                palette = ['mediumvioletred', 'palegreen', 'deepskyblue', 'coral', 'gainsboro'],
                edgecolor = 'black',
                ax = ax[1]
               )
    
    for y in [50, 75, 90, 100]:
        ax[0].axhline(y = y, linestyle = 'dashed', color = 'black')
        ax[1].axhline(y = y, linestyle = 'dashed', color = 'black')
        ax[1].text(3.6, y, f'{y}%')
    
    ax[0].set_xticklabels(ax[0].get_xticklabels(), fontdict = {"size" : "12"})
    ax[1].set_xticklabels(ax[1].get_xticklabels(), fontdict = {"size" : "12"})
    ax[0].set_xlabel('Metrics', fontdict = fontdict_labels)
    ax[1].set_xlabel('Metrics', fontdict = fontdict_labels)
    ax[0].set_ylabel('Score (%)', fontdict = fontdict_labels)
    ax[1].set_ylabel('Mean Value for Class 1 (%)', fontdict = fontdict_labels)
    ax[0].set_title('Classification Performances', fontdict = fontdict_title)
    ax[1].set_title('Cross Validation Results', fontdict = fontdict_title)
    
    handles, _ = ax[0].get_legend_handles_labels()
    legend = ax[0].legend(handles=handles, labels=['Class 0', 'Class 1'], title= None, framealpha = 1, facecolor = 'white', edgecolor = 'black')
    legend.get_title().set_fontsize('12')
    for text in legend.get_texts():
        text.set_fontsize('10')
    
    plt.show()    

    
def report_display(report):
    """
    Displays a classification report in a formatted HTML style.

    Parameters:
    - report (dict): A dictionary containing classification report metrics, typically generated by
      scikit-learn's `classification_report` function.

    Returns:
    - None
    """
    st.write(f"""
        <div class = 'all'>
            <h3>Detailed Metrics</h3>
            <h4><span style = 'color : indigo'>Class 0 (No Cardiovascular Disease):</span></h4>
            <ul>
                <li>Precision: <b>{report['0.0']['precision']:.1%}</b></li>
                <li>Recall: <b>{report['0.0']['recall']:.1%}</b></li>
                <li>f1-score: <b>{report['0.0']['f1-score']:.1%}</b></li>
                <li>support: <b>{report['0.0']['support']}</b></li>
            </ul>
            <h4><span style = 'color : indigo'>Class 1 (Cardiovascular Disease):</span></h4>
            <ul>
                <li>Precision: <b>{report['1.0']['precision']:.1%}</b></li>
                <li>Recall: <b>{report['1.0']['recall']:.1%}</b></li>
                <li>f1-score: <b>{report['1.0']['f1-score']:.1%}</b></li>
                <li>support: <b>{report['1.0']['support']}</b></li>
            </ul>
            <h4><span style = 'color : indigo'>Overall Model Performance:</span></h4>
            <ul>
                <li>Accuracy: <b>{report['accuracy']:.1%}</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
def report_table(perf_report, cv_report):
    """
    Displays a classification report in a formatted HTML table.

    Parameters:
    - report (dict): A dictionary containing classification report metrics, typically generated by
      scikit-learn's `classification_report` function.

    Returns:
    - None
    """
    st.write(f"""
        <div class = 'all'>
            <h3>Detailed Metrics</h3>
            <table style = 'border: 1px solid black'>
                <tr style = 'border: 1px solid black; color: white; background-color: grey'>
                    <th>Metric</th>
                    <th>Class 0 <br>Absence of CV disease</th>
                    <th>Class 1 <br>Presence of CV disease</th>
                    <th>Macro Average</th>
                    <th>Weighted Average</th>
                    <th style = 'border-left: 1px dashed black'>Cross-Validation<br><i>n</i>=5</th>
                </tr>
                <tr>
                    <td>Precision</td>
                    <td>{perf_report['0.0']['precision']:.1%}</td>
                    <td>{perf_report['1.0']['precision']:.1%}</td>
                    <td>{perf_report['macro avg']['precision']:.1%}</td>
                    <td>{perf_report['weighted avg']['precision']:.1%}</td>
                    <td style = 'border-left: 1px dashed black'>{np.mean(cv_report["Precision"]):.1f} ± {np.std(cv_report["Precision"]):.1f} %</td>
                </tr>
                <tr>
                    <td>Recall</td>
                    <td>{perf_report['0.0']['recall']:.1%}</td>
                    <td>{perf_report['1.0']['recall']:.1%}</td>
                    <td>{perf_report['macro avg']['recall']:.1%}</td>
                    <td>{perf_report['weighted avg']['recall']:.1%}</td>
                    <td style = 'border-left: 1px dashed black'>{np.mean(cv_report["Recall"]):.1f} ± {np.std(cv_report["Recall"]):.1f} %</td>
                </tr>
                <tr>
                    <td>f1-score</td>
                    <td>{perf_report['0.0']['f1-score']:.1%}</td>
                    <td>{perf_report['1.0']['f1-score']:.1%}</td>
                    <td>{perf_report['macro avg']['f1-score']:.1%}</td>
                    <td>{perf_report['weighted avg']['f1-score']:.1%}</td>
                    <td style = 'border-left: 1px dashed black'>{np.mean(cv_report["f1-score"]):.1f} ± {np.std(cv_report["f1-score"]):.1f} %</td>
                </tr>
                <tr>
                    <td>Support</td>
                    <td>{perf_report['0.0']['support']}</td>
                    <td>{perf_report['1.0']['support']}</td>
                    <td>{perf_report['macro avg']['support']}</td>
                    <td>{perf_report['weighted avg']['support']}</td>
                    <td style = 'border-left: 1px dashed black'>N/A</td>
                </tr>
                <tr>
                    <td>Accuracy</td>
                    <td colspan = "4">{perf_report['accuracy']:.1%}</td>
                    <td style = 'border-left: 1px dashed black'>{np.mean(cv_report["Accuracy"]):.1f} ± {np.std(cv_report["Accuracy"]):.1f} %</td>
                </tr>
            </table>
        </div>
    """, unsafe_allow_html=True)