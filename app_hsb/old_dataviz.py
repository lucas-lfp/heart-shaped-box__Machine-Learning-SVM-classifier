import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import shapiro, mannwhitneyu, chi2_contingency, spearmanr
from matplotlib.text import Text

def data_viz():
    from hsb_functions import mean_sd_range, pval_shapiro, chi2_cardio, pval_txt, mwu_cardio, chi2_var, mean_sd_1, mean_sd_0, mean_sd, pp_1, pp_0, pp, pp_3_cardio, pp_3, pp_4_cardio, pp_4

    ## Utilitaries for data viz
    fontdict_title = {'color' : 'navy', 'family' : 'Trebuchet MS', 'size' : 16, 'weight' : 'bold'}
    fontdict_labels = {'color': 'black', 'family': 'Trebuchet MS', 'size' : 14}
    palette_cardio = {"0":'lightcyan', "1": 'firebrick'}
    palette_sex = {"female": "coral", "male" : "seagreen"}

    df_raw = pd.read_csv('streamlit_hsb/cardio_train.csv', sep = ';').set_index('id')  # The purpose of this df is to keep a version of the dataset, it should therefore not be modified.

    df = pd.read_csv('streamlit_hsb/clean_cvd.csv', sep = ',')

    # Pre-setting categorical data types to str to avoid issues with sns
    df[['smoke', 'alco', 'active', 'cardio', 'cholesterol', 'gluc','ap_aha', 'lifestyle', 'healthy_ls']] = df[['smoke', 'alco', 'active', 'cardio', 'cholesterol', 'gluc','ap_aha', 'lifestyle', 'healthy_ls']].astype('str') 

    # Setting relevant data types
    df[['sex', 'smoke', 'alco', 'active', 'cardio', 'lifestyle', 'healthy_ls']] = df[['sex', 'smoke', 'alco', 'active', 'cardio', 'lifestyle', 'healthy_ls']].astype('category')
    cat_gluc_chol = pd.CategoricalDtype(categories = ["1", "2", "3"], ordered = True)
    df[['cholesterol', 'gluc']] = df[['cholesterol', 'gluc']].astype(cat_gluc_chol)
    cat_aha = pd.CategoricalDtype(categories = ["1", "2", "3", "4"], ordered = True)
    df['ap_aha'] = df['ap_aha'].astype(cat_aha)
    cat_lifestyle = pd.CategoricalDtype(categories = ["0", "1", "2", "3", "4", "5", "6", "7"], ordered = False)
    df['lifestyle'] = df['lifestyle'].astype(cat_lifestyle)

    # Creating a class "Parameter" to access parameter-related information
    class Parameter:
        def __init__(self, name, full_name, unit, mod, mod_names):
            self.name = name
            self.full_name = full_name
            self.unit = unit
            self.label = f'{self.full_name} ({self.unit})'
            self.mod = mod
            self.mod_names = mod_names

    age = Parameter("age", "Age", "years", None, None)
    sex = Parameter("sex", "Sex", None, ['female', 'male'], ['Female', 'Male'])
    height = Parameter("height", "Height", "cm", None, None)
    weight = Parameter("weight", "Weight", "kg", None, None)
    ap_hi = Parameter("ap_hi", "Systolic Blood Pressure", "mmHg", None, None)
    ap_lo = Parameter("ap_lo", "Diastolic Blood Pressure", "mmHg", None, None)
    cholesterol = Parameter("cholesterol", "Cholesterol", None, ["1", "2", "3"], ['Normal', 'Above normal', 'Well above normal'])
    gluc = Parameter("gluc", "Glucose", None, ["1", "2", "3"], ['Normal', 'Above normal', 'Well above normal'])
    smoke = Parameter("smoke", "Tobbaco", None, ["0", "1"], ['No', 'Yes'])
    alco = Parameter("alco", "Alcohol", None, ["0", "1"], ['No', 'Yes'])
    active = Parameter("active", "Physical Activity", None, ["0", "1"], ['No', 'Yes'])
    cardio = Parameter("cardio", "Cardiovascular Disease", None, ["0", "1"], ['No', 'Yes'])
    bmi = Parameter("bmi", "BMI", "kg/m²", None, None)
    ap_m = Parameter("ap_m", "Mean Blood Pressure", "mmHg", None, None)
    ap_aha = Parameter("ap_aha", "Blood Pressure Status", None, ["1", "2", "3", "4"], ['Normal', 'Elevated', 'Hypertension stage I', 'Hypertension stage II'])
    lifestyle = Parameter("lifestyle", "Lifestyle", None, ["0", "1", "2", "3", "4", "5", "6", "7"],
                        ["Non-smoker, No alcohol, Active", "Smoker", "Alcohol", "Not Active", "Smoker & Alcohol", "Smoker & Not active", "Alcohol & Not active", "Smoker & Alcohol & Not active"])
    healthy_ls = Parameter('healthy_ls', "Healthy Lifestyle", None, ["0", "1"], ["no", "yes"])
    
    # Nested navigation within the data viz section
    feature_selection = st.sidebar.selectbox("Select Feature", [
        "Conclusion", 
        "Age", 
        "Sex", 
        "BMI, Height and Weight",
        "Blood Pressure",
        "Glucose and Cholesterol",
        "Lifestyle",
        "Target Feature"
        ])
    

    

    #################
    ##### INTRO #####
    #################

    if feature_selection == "Conclusion":
        st.markdown(
            f"""
        <div class = 'all'>
            <h1>Data Analysis Conclusions</h1>
            <p class = 'intro'>
                This part of the project consisted in analysing a large dataset of <b>{df_raw.shape[0]}</b> rows and
                <b>{df_raw.shape[1]}</b> columns. Dataset was reworked to make it 
                suitable for machine learning, and relevant visualizations were produced.
            </p>
            <p class = 'intro'>
                Feature engineering led to modifying several variables, and 5 new features 
                were created. Features were analysed, so we could better understand how
                they relate with each other, especially with the feature target.
            </p>
            <h2>Major Findings</h2>
            <p>
                Several variables showed a strong correlation with the target feature: 
                prevalence of cardiovascular diseases increased with <b>age</b>, <b>BMI</b>
                and <b>blood pressure</b>, and with levels of <b>glucose</b> and <b>cholesterol</b>
                above normality.
            </p>
            <p>
                <b>Sex</b> on the other hand did not correlate with the feature target, as the
                prevalence of cardiovascular disease was very close to <b>50%</b> among 
                both males and females. Nevertheless, sex influenced several variables:
                men had higher blood pressure, while women had slightly higher levels of
                cholesterol. There was way more smokers among men and there were 
                also more likely to drink alcohol. Men were also taller than women, but
                the distribution of BMI was comparable for both sexes.
            </p>
            <h2>Insights for the Model to be built</h2>
            <p>
                The analysis has highlighted several avenues for designing an effective model:
                <ul>
                    <li><code>age</code>: aging has a major influence on health in general.
                    It could therefore be relevant to separate the training data in groups, either
                    two groups (with a cut-off around 50 years old) or even into smaller age
                    groups.</li>
                    <li><code>sex</code> did not correlate with the target feature, but did with
                    several other feature. It could be relevant to either rebalance the dataset to
                    have 50% males (<i>vs.</i> the {df['sex'].value_counts(normalize = True)['male']:.1%}
                    in the original dataset), or to train separe models for males and females.</li>
                    <li><code>height</code>: this variable alone may not bring relevant information 
                    (unlike <code>weight</code>) and could be dropped in favor of <code>bmi</code>.</li>
                    <li><code>weight</code> and <code>bmi</code>: extreme values (high values especially)
                    may not be representative, and could have a negative influence. It could be wise
                    to compare the performances of the model with and without these extreme values.</li>
                    <li><code>gluc</code> and <code>cholesterol</code>: in both case, majority of
                    subjects had normal values for these variables. Pooling modalities <i>Above normal</i>
                    and <i>Well above normal</i> could give more weight to these features. 
                    <code>gluc</code> could also be dropped as it may not bring a lot of information,
                    overshadowed by <code>cholesterol</code></li>
                    <li><code>smoke</code>: it could be wise to drop this feature as it may be
                    biaised, as explained in the <i>Lifestyle</i> section.</li>
                </ul>
            </p>
            <h2>Limits of the Analysis</h2>
            <p>
                Despite the analysis, some elements remain unclear:
                <ul>
                    <li>Age range: individuals older than 65 years old would have been expected
                    in such dataset</li>
                    <li>Sex ratio: a sex ratio of 2 females for 1 male is unexpected in this age group</li>
                    <li>The cut-off values for the glucose and cholesterol categories are not known,
                    having a numeric values for these parameters would have been more appropriate</li>
                    <li>The definition of physical activity is not known, and may be biased, limiting
                    the information brought by this variable</li>
                    <li>The definition of Cardiovascular disease is not known: there is a great
                    variety of CV diseases that come with different clinical presentations. Such
                    diversity may impair the performances of the model.</li>
                </ul>
                Some of this limits may be related to the original study protocol that lead to the production
                of this dataset. 
            </p>
        </div>
        <div class = 'all' style = 'border: 5px solid darkgreen; margin-top: 40px'>
            <p class = 'intro' style = 'padding: 25px'>
                Ultimately, a dataset suitable for machine was shaped throught this exploratory
                data analysis phase. The comprehensive analysis of the data has provided valuable 
                insights into the potential impacts of variables on the model.
                Looking ahead to the second part of the project, we will undertake specific 
                preprocessing steps, including encoding and standardization, as we prepare 
                for the development of the classification model.
            </p>
        </div>
            """, unsafe_allow_html=True
        )


    ###############
    ##### AGE #####
    ###############

    elif feature_selection == "Age":
        st.markdown(f"""
        <div class = 'all'>
            <h1>Age</h1>
            <h3>Variable Description</h3>
        </div>""", unsafe_allow_html=True)
        
        var = age
        st.markdown(f"""
            <table class = "table_1" style = "width: 75% !important">
                <tr class = "head_tr">
                    <th>Parameter</th>
                    <th>Mean ± SD</th>
                    <th>Range</th>
                    <th>Normality</th>
                </tr>
                <tr>
                    <td style = 'font-weight: bold'>{var.full_name} ({var.unit})</td>
                    <td>{mean_sd_range(df, var)[0]}</td>
                    <td>{mean_sd_range(df, var)[1]}</td>
                    <td>{pval_shapiro(df, var)}</td>
                </tr>
            </table><br>""", unsafe_allow_html=True)
            
        st.markdown('<h3>Data Visualisation</h3>', unsafe_allow_html=True)

        # Creating a df for age-related viz, that specify the age group of a subject.

        df_age = df[['age', 'cardio']]
        df_age['age_group'] = df_age.apply(lambda row:
                                        "< 45" if row['age'] < 45
                                        else "[45 - 49]" if row['age'] < 50
                                        else "[50 - 54]" if row['age'] < 55
                                        else "[55 - 59]" if row['age'] < 60
                                        else "[60 - 65]", axis = 1)

        ######FIGURE######
        fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (12,10))
        plt.subplots_adjust(hspace = 0.55, wspace = 0.3)


        # HISTPLOT: distribution of age

        sns.histplot(x = age.name, 
                    data = df, bins = 13,
                    ax = ax[0,0], 
                    color = 'silver')

        ax[0,0].axvline(x = np.mean(df[age.name]), color = 'black', linestyle = 'dashed')



        ax[0,0].text(np.mean(df[age.name])+1.25,
                1.03*(np.max(ax[0,0].get_yticks())),
                f'Mean age: {np.mean(df[age.name]):.1f} yo.',
                bbox = {'facecolor' : 'white', 'edgecolor' : 'black', 'boxstyle' : 'round'})

        ax[0,0].set_title(f'Distribution of {age.full_name}', fontdict = fontdict_title)
        ax[0,0].set_ylabel('count', fontdict = fontdict_labels)
        ax[0,0].set_xlabel(f'{age.full_name} ({age.unit})', fontdict = fontdict_labels)

        ylim = [0,9000]
        ax[0,0].set_ylim(ylim)
        ax[0,0].set_yticks(range(0,10000,1000))

        ax[0,0].set_xticks(range(39, 66, 2))

        # BOXPLOT: Comparison of age distribution regarding CV status  
        sns.boxplot(x = 'cardio',
                    y = age.name,
                    data = df,
                    ax = ax[0,1],
                    boxprops = {'edgecolor':'black'},
                    capprops = {'color': 'black'},
                    whiskerprops = {'color': 'black'},
                    medianprops = {'color': 'black'},
                    palette = palette_cardio,
                    width = 0.3)

        ax[0,1].text(0.2, np.median(df[df['cardio'] == "0"]["age"]), f'{np.median(df[df["cardio"] == "0"]["age"]):.1f}', fontsize = 12, va = "center")
        ax[0,1].text(1.2, np.median(df[df['cardio'] == "1"]["age"]), f'{np.median(df[df["cardio"] == "1"]["age"]):.1f}', fontsize = 12, va = "center")

        ax[0,1].set_yticks(range(38,68,2))
        ax[0,1].set_ylim([38,66])
        ax[0,1].set_title(f'Patients are Older than Controls', fontdict = fontdict_title)
        ax[0,1].set_ylabel(f'{age.full_name} ({age.unit})', fontdict = fontdict_labels)
        ax[0,1].set_xlabel(None)
        ax[0,1].set_xticks(ticks = ax[0,1].get_xticks(), labels = ['Controls', 'Patients'], fontsize = 14)

        # COUNTPLOT: Prevalence of CV disease in every age group


        sns.countplot(x = "age_group", 
                    hue = "cardio",
                    data = df_age, 
                    order = ["< 45", "[45 - 49]", "[50 - 54]", "[55 - 59]", "[60 - 65]"], 
                    ax = ax[1,0], 
                    edgecolor = 'black', 
                    width = 0.5,
                    palette = palette_cardio
                    )

        ax[1,0].set_title('Inequal Distribution of Patients \nAmong Age Groups', fontdict = fontdict_title)
        ax[1,0].set_ylabel('Count', fontdict = fontdict_labels)
        ax[1,0].set_xlabel('Age Groups (years)', fontdict = fontdict_labels)
        ax[1,0].set_yticks(range(0, 11000, 1000))

        handles = [
            Patch(facecolor = palette_cardio["0"], edgecolor = 'black',  label = "Controls"),
            Patch(facecolor = palette_cardio["1"], edgecolor = 'black',  label = "Patients")
        ]

        ax[1,0].legend(handles = handles, title = None, frameon = True, edgecolor = 'black')


        # LINEPLOT: Prevalence of CV disease according to age
        df_age['age'] = df_age['age'].astype('int64')
        df_age.groupby('age').value_counts(normalize = True)
        df_age_gb = df_age[['age', 'cardio']].groupby('age')['cardio'].value_counts(normalize = True).unstack().reset_index()


        sns.lineplot(x = "age", 
                    y = "1", 
                    data = df_age_gb[df_age_gb["age"] >=39],
                    ax = ax[1,1],
                    color = 'purple',
                    linewidth = 3
                    )

        ax[1,1].axhline(y = 0.5, linestyle = "dotted", color = 'black')

        ax[1,1].set_title('Cardiovascular Disease Prevalence \nIncreases with Age', fontdict = fontdict_title)
        ax[1,1].set_ylabel('CVD Prevalence (%)', fontdict = fontdict_labels)
        ax[1,1].set_yticks(ticks = [x/10 for x in range(0,11,1)], labels = range(0,110,10))
        ax[1,1].set_xlabel(f'{age.full_name} ({age.unit})', fontdict = fontdict_labels)
        ax[1,1].set_xticks(ticks = range(39,66,2))

        st.pyplot(fig)

        st.markdown(f"""
        <div class = 'all'>
            <h3>Variable Analysis</h3>
            <p>
                The variable <code>{age.name}</code> is not normally distributed and shows a left skew. The minimal age value found
                in the original dataset was {np.min(df_raw['age'])/365.25:.1f} yo. There was {len(df_raw[df_raw['age']/365.25 < 32])}
                individuals below 32 years old, and there no subject in the age group [32 - 38] yo. None 
                of the subjects below 32 years old had cardiovascular disease, and they were considered outliers due to the existing age gap. It was decided to
                take them out of the analysis. Consequently, the range for <code>{age.name}</code> is
                <b>[{np.min(df[age.name]):.0f} - {np.max(df[age.name]):.0f}]</b> years old.
                The max age value is surprisingly low as cardiovascular diseases are common among older individuals,
                the reason why no data from elderly patients were recorded is not known, but was probably a protocol requirement.
            </p>
            <p>
                The prevalence of CV diseases is known to increase with age, as it is the case in this cohort. The proportion of
                patients in the age group [51 - 55] is <b>around 50%</b>. It is below 50% in younger individuals and above in older individuals.
                The mean age in patients is <b>{mean_sd_range(df[df['cardio'] == "1"], age)[0]}</b> years old compared to
                <b>{mean_sd_range(df[df['cardio'] == "0"], age)[0]}</b> years old in controls.
            </p>
        </div>
        <div class = 'all conclusion'>
            <h3>Conclusion</h3>
            <ul>
                <li>The mean age in the cohort was {mean_sd_range(df, age)[0]} yo.</li>
                <li>Age range was {mean_sd_range(df, age)[1]} yo: no children and no elderly people were included</li>
                <li>Patients in the cohort were older than controls, and the prevalence of CV disease linearly increased with age</li>
                <li>It could be worth adjusting the model on age, <i>ie</i> separating the data in two distinct dataframes with a cut-off value for age nearby 50 yo.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    ###############
    ##### SEX #####
    ###############

    elif feature_selection == 'Sex':
        st.write(f"""
        <div class = 'all'>
            <h1>Sex</h1>
            <p class = 'intro'>
                Note: This section specifically focuses on the relationship between sex and 
                age, as well as sex and cardiovascular disease prevalence. 
                Influence of sex on other variables is assessed in their dedicated sections.
            </p>
            <h3>Variable Description</h3>
        </div>
        """, unsafe_allow_html=True)

        var = sex

        table = f"""
        <table class = "table_1" style = "width: 75% !important">
            <tr class = "head_tr">
                <th>Parameter</th>
                <th>Modalities</th>
                <th>Observations (<i>n</i>)</th>
                <th>Observations (%)</th>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td rowspan = "2" style = 'font-weight: bold'>{var.full_name}</td>
                <td style = 'border-left: 1px dashed black'>{var.mod[0]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var.name].value_counts()[var.mod[0]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var.name].value_counts(normalize = True)[var.mod[0]]:.1%}</td>
            </tr>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td style = 'border-left: 1px dashed black'>{var.mod[1]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var.name].value_counts()[var.mod[1]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var.name].value_counts(normalize = True)[var.mod[1]]:.1%}</td>
            </tr>
        </table><br>"""
        
        st.write(table, unsafe_allow_html=True)    
        st.write('<h3>Data Visualisation</h3>', unsafe_allow_html=True)

        ###### FIGURE ######

        df_age_m = df[df['sex'] == 'male'][['age', 'cardio']]
        df_age_m['age_group'] = df_age_m.apply(lambda row:
                                        "< 45" if row['age'] < 45
                                        else "[45 - 49]" if row['age'] < 50
                                        else "[50 - 54]" if row['age'] < 55
                                        else "[55 - 59]" if row['age'] < 60
                                        else "[60 - 65]", axis = 1)

        df_age_m['age'] = df_age_m['age'].astype('int64')
        df_age_m.groupby('age').value_counts(normalize = True)
        df_age_gb_m = df_age_m[['age', 'cardio']].groupby('age')['cardio'].value_counts(normalize = True).unstack().reset_index()

        df_age_f = df[df['sex'] == 'female'][['age', 'cardio']]
        df_age_f['age_group'] = df_age_f.apply(lambda row:
                                        "< 45" if row['age'] < 45
                                        else "[45 - 49]" if row['age'] < 50
                                        else "[50 - 54]" if row['age'] < 55
                                        else "[55 - 59]" if row['age'] < 60
                                        else "[60 - 65]", axis = 1)

        df_age_f['age'] = df_age_f['age'].astype('int64')
        df_age_f.groupby('age').value_counts(normalize = True)
        df_age_gb_f = df_age_f[['age', 'cardio']].groupby('age')['cardio'].value_counts(normalize = True).unstack().reset_index()

        fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (12, 10))
        plt.subplots_adjust(hspace = 0.3)

        pie_females = df['sex'].value_counts(normalize = True)['female']
        pie_males = df['sex'].value_counts(normalize = True)['male']

        ax[0,0].pie(x = [pie_females, pie_males], 
                labels = ['Females', 'Males'], 
                colors = ['coral', 'seagreen'], 
                autopct = '%1.1f%%', 
                startangle = 90, 
                wedgeprops = {'ec' : 'black'}, textprops = {'size' : 14, 'weight' : 'bold'})
        ax[0,0].set_title('Distribution of Sex', fontdict = fontdict_title)

        # BARPLOTS

        df_sex = df.groupby('sex')['cardio'].value_counts(normalize = True).unstack().reset_index().rename(columns = {'0': 'cardio_0', '1': 'cardio_1'})
        df_sex_cardio = pd.melt(df_sex, id_vars = "sex", var_name = "cardio_type", value_name = "prevalence")

        sns.barplot(
            x = 'sex',
            y = 'prevalence',
            hue = 'cardio_type',
            data = df_sex_cardio,
            ax = ax[0,1],
            order = ['female', 'male'],
            edgecolor = 'black',
            palette = [palette_cardio["0"], palette_cardio["1"]]
            )
        
        for container in ax[0,1].containers:
            for bar in container:
                x = bar.get_x()
                y = bar.get_height()
                ax[0,1].text(x + 0.21, y + 0.02, f'{y:.1%}', weight = 'bold', size = 12, ha = "center")

        ylim = [0,0.8]
        ax[0,1].set_ylim(ylim)
        ax[0,1].set_yticks(ticks = [x/10 for x in range(0,9,1)], labels = range(0,90,10))

        ax[0,1].set_yticks(ticks = ax[0,1].get_yticks())
        ax[0,1].set_ylabel('Subjects (%)', fontdict = fontdict_labels)

        ax[0,1].set_xticks(ticks = ax[0,1].get_xticks(), labels = ['Females', 'Males'], size = 12)
        ax[0,1].set_xlabel(None)

        ax[0,1].set_title(f'Sex May Not Influence Cardiovascular \nDisease Prevalence', fontdict = fontdict_title)

        # Custom legend
        handles = [
            Patch(facecolor = palette_cardio["0"], edgecolor = 'black', label = 'Controls'),
            Patch(facecolor = palette_cardio["1"], edgecolor = 'black', label = 'Patients')
        ]

        ax[0,1].legend(
            handles = handles,
            edgecolor = 'black',
            title = None
            )


        ### Age Boxplot

        sns.boxplot(x = "sex", 
                    y = "age", 
                    data = df, 
                    palette = palette_sex,
                    color = 'black',
                    ax = ax[1,0],
                    width = 0.3,
                    boxprops={'edgecolor':'black'},
                    capprops={'color':'black'},
                    medianprops={'color':'black'},
                    flierprops={'color':'black'},
                    whiskerprops={'color':'black'}
                )

        mean_age_fem = np.mean(df[df['sex'] == 'female']['age'])
        mean_age_mal = np.mean(df[df['sex'] == 'male']['age'])
                            
        ax[1,0].text(0.2, round(mean_age_fem,1), f'{mean_age_fem:.1f}', fontweight = "bold", ha = "left")
        ax[1,0].text(1.2, round(mean_age_mal,1), f'{mean_age_mal:.1f}', fontweight = "bold", ha = "left")

        ax[1,0].set_ylim([30,70])
        ax[1,0].set_yticks(ticks = range(30,75,5), labels = range(30,75,5))
        ax[1,0].set_ylabel("Age (years)", fontdict = fontdict_labels)

        ax[1,0].set_xticks(ticks = ax[1,0].get_xticks(), labels = ["Females", "Males"], fontsize = 12)
        ax[1,0].set_xlabel(None)

        ax[1,0].set_title("Comparable Distribution of Age \nwas found in both Groups", fontdict = fontdict_title)


        # LINEPLOT

        sns.lineplot(x = "age", 
                    y = "1", 
                    data = df_age_gb_f,
                    ax = ax[1,1],
                    color = palette_sex['female'],
                    linewidth = 3
                    )

        sns.lineplot(x = "age", 
                    y = "1", 
                    data = df_age_gb_m,
                    ax = ax[1,1],
                    color = palette_sex['male'],
                    linewidth = 3
                    )

        ax[1,1].axhline(y = 0.5, linestyle = "dotted", color = 'black')

        ax[1,1].set_title('CVD Prevalence Increases with Age \nRegardless of Sex', fontdict = fontdict_title)
        ax[1,1].set_ylabel('Prevalence (%)', fontdict = fontdict_labels)
        ax[1,1].set_yticks(ticks = [x/10 for x in range(0,11,1)], labels = range(0,110,10))
        ax[1,1].set_xlabel(f'{age.full_name} ({age.unit})', fontdict = fontdict_labels)
        ax[1,1].set_xticks(ticks = range(39,66,2))

        handles = [
            Line2D([0], [0], color = palette_sex['female'], label = 'Females', marker = None),
            Line2D([0], [0], color = palette_sex['male'], label = 'Males', marker = None)
        ]

        ax[1,1].legend(
            handles = handles,
            title = None,
            edgecolor = 'black'
        )

        st.pyplot(fig)

        st.write(f"""
        <div class = 'all'>
            <h3>Variable Analysis</h3>
            <p>
                <code>sex</code> showed an important degree of imbalance, with roughly <b><sup>2</sup>/<sub>3</sub>
                of women</b> and <sup>1</sup>/<sub>3</sub> of men. The reasons
                for such an imbalance remain unclear in this age group, as it is closer to the expected sex ratio in a geriatric cohort.
            </p>
            <p>
                The distribution of cardiovascular disease among men and women is visually balanced, with a prevalence of
                <b>{df[df['sex'] == 'female']['cardio'].value_counts(normalize = True)["1"]:.1%}</b> among women and of
                <b>{df[df['sex'] == 'male']['cardio'].value_counts(normalize = True)["1"]:.1%}</b> among men. CV disease was not
                linked to sex according to chi² contingency test ({chi2_cardio(df, sex)}). Nevertheless, it should be kept in mind that having
                comparable proportion of Controls and Patients in each sex group might have been a protocol criteria, in which
                case this observation would be biased.
            </p>
            <p>
                Age distribution was comparable in both sex groups, with female having a mean age of <b>{np.mean(df[df['sex'] == 'female']['age']):.1f} ±
                {np.std(df[df['sex'] == 'female']['age']):.1f}</b> years old <i>vs.</i> 
                <b>{np.mean(df[df['sex'] == 'male']['age']):.1f} ± {np.std(df[df['sex'] == 'male']['age']):.1f}</b> years old in males. 
                Moreover, prevalence of cardiovascular disease increased with age in both sex group was superimposable.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.write(f"""
        <div class = 'all conclusion'>
            <h3>Conclusion</h3>
            <ul>
                <li>No obvious influence of sex was observed, as CV disease prevalence was very close to 50% in both groups</li>
                <li>Prevalence of CV diseases increased with age in a comparable manner in both males and females</li>
                <li>As there is almost twice as much females in the cohort, we may consider rebalancing the training data to a 1:1 sex
                ratio or train distinct models for males and females</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    ###############################
    ##### BMI, HEIGHT, WEIGHT #####
    ###############################

    elif feature_selection == "BMI, Height and Weight":
        st.write(f"""
        <div class = 'all'>
            <h1>BMI, Height and Weight</h1>
            <h3>Variable Description</h3>
        </div>
        """, unsafe_allow_html=True)

        var = [bmi, height, weight]
        st.write(f"""
            <table class = "table_1" style = "width: 75%">
                <tr class = "head_tr">
                    <th>Parameter</th>
                    <th>Mean ± SD</th>
                    <th>Range</th>
                    <th>Normality</th>
                </tr>
                <tr>
                    <td style = 'font-weight: bold'>{var[0].full_name} ({var[0].unit})</td>
                    <td>{mean_sd_range(df, var[0])[0]}</td>
                    <td>{mean_sd_range(df, var[0])[1]}</td>
                    <td>{pval_shapiro(df, var[0])}</td>
                </tr>
                <tr style = 'border-top: 1.5px solid black'>
                    <td style = 'font-weight: bold'>{var[1].full_name} ({var[1].unit})</td>
                    <td>{mean_sd_range(df, var[1])[0]}</td>
                    <td>{mean_sd_range(df, var[1])[1]}</td>
                    <td>{pval_shapiro(df, var[1])}</td>
                </tr>
                <tr style = 'border-top: 1.5px solid black'>
                    <td style = 'font-weight: bold'>{var[2].full_name} ({var[2].unit})</td>
                    <td>{mean_sd_range(df, var[2])[0]}</td>
                    <td>{mean_sd_range(df, var[2])[1]}</td>
                    <td>{pval_shapiro(df, var[2])}</td>
                </tr>
            </table><br>""", unsafe_allow_html=True)

        # Creating a df with rounded bmi and aggregated lower and upper values for plotting purpose

        df['round_bmi'] = df.apply(lambda row: 45 if round(row['bmi'],0) > 40 else 
                                15 if round(row['bmi'],0) < 18 else round(row['bmi'],0), axis = 1)
        df_bmi = df.groupby('round_bmi')['cardio'].value_counts(normalize = True).unstack().reset_index()


        st.write('<h3>Data Visualisation</h3>', unsafe_allow_html=True)

        ######FIGURE######
        fig = plt.figure(figsize = (10, 15))
        fig.subplots_adjust(wspace = 0.3, hspace = 0.4)
        spec = gridspec.GridSpec(nrows = 3, ncols = 2)

        # HISTPLOT bmi
        ax = fig.add_subplot(spec[0, 0])
        sns.histplot(x = 'bmi', data = df, ax = ax, color = 'silver', bins = 20)
        ax.set_title('Distribution of BMI', fontdict = fontdict_title)
        ax.set_xticks(range(0, int(np.max(ax.get_xticks())), 5))
        ax.set_ylabel('Count', fontdict = fontdict_labels)
        ax.set_xlabel('BMI (kg/m²)', fontdict = fontdict_labels)
        ax.set_xlim([10,90])
        ax.set_xticks(ticks = range(10,95,5), labels = range(10,95,5))
        ax.set_ylim([0, 22500])
        ax.set_yticks(ticks = range(0,25000,2500))

        # LINEPLOT CV disease and BMI
        ax = fig.add_subplot(spec[0, 1])
        sns.lineplot(
            x = 'round_bmi',
            y = '1', 
            data = df_bmi[~df_bmi['round_bmi'].isin([15,45])], 
            color = 'purple', 
            linewidth = 3
        )

        sns.lineplot(
            x = 'round_bmi',
            y = '1', 
            data = df_bmi[df_bmi['round_bmi'].isin([15,18])], 
            color = 'purple', 
            linewidth = 3,
            linestyle = "dotted"
        )

        sns.lineplot(
            x = 'round_bmi',
            y = '1', 
            data = df_bmi[df_bmi['round_bmi'].isin([40,45])], 
            color = 'purple', 
            linewidth = 3,
            linestyle = "dotted"
        )

        ax.axhline(y = 0.5, linestyle = 'dotted', color = 'black')

        handles = [
            Line2D([0], [0], label = "Pooled data from individual \nwith extreme BMI, displaying\nhigh variability", linestyle = "dotted", color = "purple")
        ]

        ax.legend(handles = handles, title = None, edgecolor = "black")

        ax.set_ylim([0,1])
        ax.set_yticks(ticks = [x/10 for x in range(11)], labels = range(0,110,10))
        ax.set_ylabel('CVD Prevalence (%)', fontdict = fontdict_labels)
        
        ax.set_title('Cardiovascular Disease Prevalence \nIncreases with BMI', fontdict = fontdict_title)
        
        ax.set_xlim([13,47])
        ax.set_xticks(ticks = range(15,50,5), labels = ["< 17", 20, 25, 30, 35, 40, "> 40"])
        ax.set_xlabel(bmi.label, fontdict = fontdict_labels)

        # BMI distribution according to sex

        ax = fig.add_subplot(spec[1,0])

        sns.boxplot(
            x = "sex", 
            y = "bmi", 
            data = df,
            palette = palette_sex,
            width = 0.3,
            medianprops = {'color' : 'black'},
            boxprops = {'edgecolor': 'black'},
            flierprops = {'color' : 'black', 'markersize': 3},
            whiskerprops = {'color' : 'black'},
            capprops = {'color' : 'black'}
        )

        median_bmi_f = np.median(df[df['sex'] == 'female']['bmi'])
        median_bmi_m = np.median(df[df['sex'] == 'male']['bmi'])

        ax.text(0.3, median_bmi_f, f'{median_bmi_f:.1f}', ha = "center", va = "center", fontweight = "bold")
        ax.text(1.3, median_bmi_m, f'{median_bmi_m:.1f}', ha = "center", va = "center", fontweight = "bold")

        ax.set_ylim([0,90])
        ax.set_yticks(ticks = range(0,100,10), labels = range(0,100,10))
        ax.set_ylabel('BMI (kg/m²)', fontdict = fontdict_labels)

        ax.set_xlabel(None)
        ax.set_xticks(ticks = ax.get_xticks(), labels = ['Females', 'Males'], fontsize = 12)

        ax.set_title("Sex has Limited Impact on BMI", fontdict = fontdict_title)

        # SCATTERPLOT height weight sex
        ax = fig.add_subplot(spec[1,1])
        sns.scatterplot(x= 'weight',
                        y = 'height',
                        data = df,
                        ax = ax,
                        edgecolor = 'black',
                        hue = 'sex',
                        alpha = 0.5,
                        palette = palette_sex,
                        s = 10)

        h_lo = np.percentile(df['height'], 2.5)
        h_hi = np.percentile(df['height'], 97.5)
        w_lo = np.percentile(df['weight'], 2.5)
        w_hi = np.percentile(df['weight'], 97.5)

        ax.axvline(w_lo, linestyle = 'dotted', color = 'black')
        ax.axvline(w_hi, linestyle = 'dotted', color = 'black')
        ax.axhline(h_lo, linestyle = 'dotted', color = 'black')
        ax.axhline(h_hi, linestyle = 'dotted', color = 'black')
        ax.plot([w_lo, w_hi, w_hi, w_lo, w_lo], [h_lo, h_lo, h_hi, h_hi, h_lo], color = 'black')

        ax.set_title('High Variability Found for Height \nand Weight Extreme Values', fontdict = fontdict_title)
        ax.set_ylabel('Weight (kg)', fontdict = fontdict_labels)

        xlim = [40, 220]
        ax.set_xlim(xlim)
        ax.set_xticks(ticks = range(40,240,20), labels = range(40,240,20))
        ax.set_xlabel('Weight (kg)', fontdict = fontdict_labels)

        ax.set_ylabel('Height (cm)', fontdict = fontdict_labels)
        ax.set_ylim([120, 220])
        ax.set_yticks(ticks = range(120,230,10), labels = range(120,230,10))

        ax.legend(title = 'Sex', frameon = True, edgecolor ='black')

        ax_range = xlim[1] - xlim[0]
        ax.text(w_lo + 0.01*ax_range, 121, f'{w_lo:.0f} kg')
        ax.text(w_hi + 0.01*ax_range, 121, f'{w_hi:.0f} kg')
        ax.text(xlim[1]-1, h_lo+1, f'{h_lo:.0f} cm', ha = "right")
        ax.text(xlim[1]-1, h_hi+1, f'{h_hi:.0f} cm', ha = "right")

        handles = [
            Line2D([0], [0], color = 'w', markerfacecolor = palette_sex['female'], label = 'Female', marker = 'o', markeredgecolor = 'black'),
            Line2D([0], [0], color = 'w', markerfacecolor = palette_sex['male'], label = 'Male', marker = 'o', markeredgecolor = 'black'),
            Line2D([0], [0], color = 'black', linestyle = 'dotted', label = '95% CI')
        ]

        ax.legend(handles = handles, edgecolor = 'black')

        # HISTPLOT focus on outliers
        ax = fig.add_subplot(spec[2,:])

        df["out_wh"] = df.apply(lambda row:
                                "hi_wh" if row['weight'] > w_hi and row['height'] > h_hi
                                else "hi_w" if row['weight'] > w_hi
                                else "hi_h" if row['height'] > h_hi
                                else "lo_wh" if row['weight'] < w_lo and row['height'] < h_lo
                                else "lo_w" if row['weight'] < w_lo
                                else "lo_h" if row['height'] < h_lo
                                else "normal", axis = 1
                            )

        df_wh_temp = df.groupby('out_wh')['cardio'].value_counts(normalize = True).unstack().reset_index()
        df_wh = pd.melt(df_wh_temp, id_vars = "out_wh", var_name = "cardio", value_name = "proportion") 

        sns.barplot(x = 'out_wh', 
                    y ="proportion",
                    data = df_wh, 
                    hue = "cardio", 
                    ax =ax, 
                    edgecolor = 'black', 
                    order = ['hi_wh', 'hi_w', 'hi_h', 'lo_wh', 'lo_w', 'lo_h', 'normal'],
                    palette = palette_cardio
                )

        ax.axhline(y = 0.5, linestyle = 'dashed', color = 'black')

        for index, group in enumerate(['hi_wh', 'hi_w', 'hi_h', 'lo_wh', 'lo_w', 'lo_h', 'normal']):
            ax.text(index, 0.1, f'n = {len(df[df["out_wh"] == group])}', ha ="center", bbox = {'facecolor' : 'white', 'edgecolor' : 'black', 'boxstyle' : 'round'})

        ax.set_title('Extreme Weight and/or Height May Influence \nCardiovascular Disease Prevalence', fontdict = fontdict_title)

        ylim = ([0,1])
        ax.set_ylim(ylim)
        ax.set_yticks(ticks = [x/10 for x in range(11)], labels = range(0,110,10))
        ax.set_ylabel("Subjects (%)", fontdict = fontdict_labels)

        ax.set_xlabel(None)
        ax.set_xticklabels(['High Weight\n and Height', 'High Weight', 'High Height', 'Low Weight\nand Height', 'Low Weight', 'Low Height', 'Normal Weight\nand Height'])

        handles = [
            Patch(facecolor = palette_cardio["0"], label = "Controls", edgecolor = "black"),
            Patch(facecolor = palette_cardio["1"], label = "Patients", edgecolor = "black")
        ]

        ax.legend(handles = handles, title = None, edgecolor = "black")

        st.pyplot(fig)

        st.write(f"""
        <div class ='all'>
            <h3>Variables Analysis</h3>
            <p>
                A wide range of values was covered by both <code>height</code> ({mean_sd_range(df, height)[1]} cm) and <code>weight</code>
            ({mean_sd_range(df, weight)[1]} kg). Consequently, range for <code>bmi</code> was <b>{mean_sd_range(df, bmi)[1]}</b> kg/m².
            The median values for <code>bmi</code> was <b>{np.median(df['bmi']):.1f}</b> kg/m². Men were taller than women, 
            but the distribution of BMI was comparable in both groups.
            </p>
            <p>
                Data suggest that the <b>prevalence of cardiovascular diseases increases
            with BMI</b>. Regarding weight and height, extreme values showed great variability; for both high and low values of height 
            and high values only for weight.
            </p>
            <p>
                The prevalence of cardiovascular disease was around <b>50%</b> among subjects with both height and weight within their
            respective 95% confidence interval ([{h_lo} - {h_hi}] cm for height and [{w_lo} - {w_hi}] kg for weight). On extreme
            values subgroups, weight seemed to have a greater influence on CV disease than height.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.write(f"""
        <div class = 'all conclusion'>
            <h3>Conclusion</h3>
            <ul>
                <li>Cardiovascular diseases prevalence increases with BMI</li>
                <li>Distribution of <code>bmi</code> is right-skewed, with a max value of {np.max(df['bmi']):.1f} kg/m², but outlying
                values may have a limited impact</li>
                <li>Among subjects with outlying values for <code>height</code> and/or <code>weight</code>, weight seemed to have
                a greater influence than height</li>
                <li>It may be relevant to assess the influence of extreme values for <code>height</code> and <code>weight</code> on the 
                model</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    ##########################
    ##### BLOOD PRESSURE #####
    ##########################

    elif feature_selection == "Blood Pressure":

        st.write(f"""
            <div class = 'all'>
                <h1>Blood Pressure</h1>
                <h3>Variables Description</h3>
            </div>
            """, unsafe_allow_html=True)


        var = [ap_hi, ap_lo, ap_m]

        st.write(f"""
            <table class = "table_1" style = "width: 75% !important">
                <tr class = "head_tr">
                    <th>Parameter</th>
                    <th>Mean ± SD</th>
                    <th>Range</th>
                    <th>Normality</th>
                </tr>
                <tr>
                    <td style = 'font-weight: bold'>{var[0].full_name} ({var[0].unit})</td>
                    <td>{mean_sd_range(df, var[0])[0]}</td>
                    <td>{mean_sd_range(df, var[0])[1]}</td>
                    <td>{pval_shapiro(df, var[0])}</td>
                </tr>
                <tr>
                    <td style = 'font-weight: bold'>{var[1].full_name} ({var[1].unit})</td>
                    <td>{mean_sd_range(df, var[1])[0]}</td>
                    <td>{mean_sd_range(df, var[1])[1]}</td>
                    <td>{pval_shapiro(df, var[1])}</td>
                </tr>
                <tr>
                    <td style = 'font-weight: bold'>{var[2].full_name} ({var[2].unit})</td>
                    <td>{mean_sd_range(df, var[2])[0]}</td>
                    <td>{mean_sd_range(df, var[2])[1]}</td>
                    <td>{pval_shapiro(df, var[2])}</td>
                </tr>
            </table><br>""", unsafe_allow_html=True)

        var = ap_aha
        table = f"""
        <table class = "table_1" style = "width: 75% !important">
            <tr class = "head_tr">
                <th>Parameter</th>
                <th>Modalities</th>
                <th>Observations (<i>n</i>)</th>
                <th>Observations (%)</th>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td rowspan = "4" style = 'font-weight: bold'>{var.full_name}</td>
                <td style = 'border-left: 1px dashed black'>{var.mod_names[0]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var.name].value_counts()[var.mod[0]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var.name].value_counts(normalize = True)[var.mod[0]]:.1%}</td>
            </tr>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td style = 'border-left: 1px dashed black'>{var.mod_names[1]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var.name].value_counts()[var.mod[1]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var.name].value_counts(normalize = True)[var.mod[1]]:.1%}</td>
            </tr>   
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td style = 'border-left: 1px dashed black'>{var.mod_names[2]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var.name].value_counts()[var.mod[2]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var.name].value_counts(normalize = True)[var.mod[2]]:.1%}</td>
            </tr>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td style = 'border-left: 1px dashed black'>{var.mod_names[3]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var.name].value_counts()[var.mod[3]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var.name].value_counts(normalize = True)[var.mod[3]]:.1%}</td>
            </tr>
        </table><br>"""
        
        st.write(table, unsafe_allow_html=True)    

        st.write(f"""
        <div class = 'all'>
            <h4>Several issues were noted while exploring blood pressure data.</h4>
            <p>             
                <ul>
                    <li>Negative values: A careful examination of these observations suggested that the negative sign had been mistakenly entered,
                    as the absolute values were plausible values for systolic blood pressure (e.g., -150, -115, etc.) 
                    or diastolic blood pressure values (-70).</li>
                    <li>Zeros: There were 17 observation with the value 0 for <code>ap_lo</code>, all were considered missing values.</li>
                    <li>Extremely low values: Regarding extremely low values, mainly between 11 
                    and 15 for <code>ap_hi</code> and between 5 and 10 for <code>ap_lo</code>. It is very likely that these 
                    values were entered in cmHg and not mmHg.</li>
                    <li>Extremely high values: About 30 observations had extremely high values for <code>ap_hi</code>, again with patterns suggesting an issue
                    with units, or that both systolic and diastolic were entered in the same column.</li>
                    <li>Inconsistency between both features: In some cases, <code>ap_hi</code> was lower than <code>ap_lo</code>.
                    A carefull review of said observations revealed that both values were clinically relevant, and were swapped back to
                    the right column.</li>
                </ul>
                Negative values, zeros, extremely high and low values were all dropped, considering that the hypotheses to explain these
                values could not be verified. In addition, regarding the small number of affected observations compared to the dataset's 
                length, dropping these rows won't not impact the model, so it was safer to drop rows that could contain false values.
                Finally, regarding extreme values, it was arbitrarily decided to fix cut-off values of 60 and 300 for <code>ap_hi</code>
                and 40 and 250 for <code>ap_lo</code>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.write('<h3>Data Visualisation</h3>', unsafe_allow_html=True)

        df_ap = df[['age', 'sex', 'ap_hi', 'ap_lo', 'ap_m', 'ap_aha', 'cardio']]
        df_ap['age'] = df_ap['age'].astype('int64')

        df_ap['age_5'] = df_ap.apply(lambda row: 40 if row['age'] < 45
                                    else 45 if row['age'] < 50
                                    else 50 if row['age'] < 55
                                    else 55 if row['age'] < 60
                                    else 60 if row['age'] < 65
                                    else 65, axis = 1)

        df_ap_all = df_ap.groupby('age_5')['ap_m'].mean().reset_index()
        df_ap_fem_0 = df_ap[(df_ap['sex'] == 'female') & (df_ap['cardio'] == "0")].groupby('age_5')['ap_m'].mean().reset_index()
        df_ap_fem_1 = df_ap[(df_ap['sex'] == 'female') & (df_ap['cardio'] == "1")].groupby('age_5')['ap_m'].mean().reset_index()

        df_ap_mal_0 = df_ap[(df_ap['sex'] == 'male') & (df_ap['cardio'] == "0")].groupby('age_5')['ap_m'].mean().reset_index()
        df_ap_mal_1 = df_ap[(df_ap['sex'] == 'male') & (df_ap['cardio'] == "1")].groupby('age_5')['ap_m'].mean().reset_index()

        ######FIGURE######
        fig  = plt.figure(figsize = (10,15))
        spec = gridspec.GridSpec(nrows = 3, ncols = 2)

        # HISTPLOT systolic and diastolic distribution
        ax = fig.add_subplot(spec[0,0])
        sns.histplot(
            data = df[['ap_hi', 'ap_lo']], 
            ax=ax,
            bins = 18, 
            palette = ['gold', 'indigo'])

        ax.set_xticks(range(40, int(np.max(ax.get_xticks())), 20))
        ax.set_title('Distribution of Systolic \nand Diastolic Blood Pressure', fontdict = fontdict_title)
        ax.set_xlabel('Blood Pressure (mmHg)', fontdict = fontdict_labels)
        ax.set_ylabel('Count', fontdict = fontdict_labels)
        
        handles = [
            Patch(facecolor = 'khaki', edgecolor = 'black', label = 'Systolic'),
            Patch(facecolor = 'darkorchid', edgecolor = 'black', label = 'Diastolic')
        ]

        ax.legend(handles = handles, title = None, edgecolor = 'black')

        # SCATTERPLOT systolic and diastolic correlation
        ax = fig.add_subplot(spec[0,1])
        sns.scatterplot(
            x = 'ap_hi',
            y = 'ap_lo', 
            data = df, 
            ax = ax, 
            hue = 'cardio',
            palette = palette_cardio, 
            edgecolor = 'black', 
            alpha = 1, 
            hue_order = ["0", "1"], 
            )


        ax.set_title('Higher Blood Pressure Values \nin Patients', fontdict = fontdict_title)
        ax.set_xticks(range(40,250,20))
        ax.set_yticks(range(40,250,20))
        ax.set_xlabel('Systolic pressure (mmHg)', fontdict = fontdict_labels)
        ax.set_ylabel('Diastolic pressure (mmHg)', fontdict = fontdict_labels)

        handles = [
            Line2D([0], [0], color = 'white', marker = 'o', markerfacecolor= palette_cardio["0"], markeredgecolor='black', label = 'Controls'),
            Line2D([0], [0], color = 'white', marker = 'o', markerfacecolor= palette_cardio["1"], markeredgecolor='black', label = 'Patients')
        ]
        ax.legend(title = None, handles = handles, edgecolor = 'black')


        # Prevalence of CV disease according to ap_aha
        ax = fig.add_subplot(spec[1,0])

        cardio_aha = (df.groupby('ap_aha')['cardio'].value_counts(normalize = True)*100).unstack().reset_index().rename(columns = {"0" : "cardio_0", "1" : "cardio_1"})

        sns.lineplot(
            x = "ap_aha", 
            y = "cardio_1", 
            data = cardio_aha, 
            linewidth = 3, 
            ax = ax, 
            color = "purple", 
            marker = "o",
            markeredgecolor = 'black'
            )

        for i in range(1, 5, 1):
            prevalence = cardio_aha[cardio_aha['ap_aha'] == str(i)]['cardio_1'].values[0]
            ax.text(i-1, prevalence + 4, f'{prevalence:.1f}%', fontweight = 'bold', ha = "right")


        ax.set_ylim(0,100)
        ax.set_yticks(range(0,110,10))
        ax.set_ylabel("CVD Prevalence (%)", fontdict = fontdict_labels)

        ax.set_xticks(ticks = ax.get_xticks(), labels = ["Normal", "Elevated", "Hypertension \nStage 1", "Hypertension \nStage 2"])
        ax.set_xlabel("Blood Pressure Classification \n(according to AHA)", fontdict = fontdict_labels)
        ax.set_xlim(-0.5,3.3)
        ax.set_title("Higher Prevalence of CV Disease in AHA's \nCategories Hypertension Stage 1 & 2", fontdict = fontdict_title)

        # LINEPLOT ap as a function of age, sex and CV status
        ax = fig.add_subplot(spec[1, 1])

        sns.lineplot(
            x= 'age_5', 
            y= 'ap_m', 
            data = df_ap_fem_0, 
            color = palette_sex['female'], 
            linestyle = 'dotted', 
            linewidth = 3, 
            marker = 'o', 
            markeredgecolor = 'black'
            )
        
        sns.lineplot(
            x= 'age_5',
            y= 'ap_m', 
            data = df_ap_fem_1, 
            color = palette_sex['female'], 
            linestyle = 'solid', 
            linewidth = 3, 
            marker = 'o', 
            markeredgecolor = 'black'
            )

        sns.lineplot(
            x= 'age_5', 
            y= 'ap_m', 
            data = df_ap_mal_0, 
            color = palette_sex['male'], 
            linestyle = 'dotted', 
            linewidth = 3, 
            marker = 'o', 
            markeredgecolor = 'black'
            )
        
        sns.lineplot(
            x= 'age_5', 
            y= 'ap_m', 
            data = df_ap_mal_1, 
            color = palette_sex['male'], 
            linestyle = 'solid', 
            linewidth = 3, 
            marker = 'o', 
            markeredgecolor = 'black'
            )
        
        ax.set_ylim(86, 104)
        ax.set_yticks(ticks = range(86,105,1), labels = range(86,105,1))
        ax.set_ylabel("Mean Blood Pressure (mmHg)", fontdict = fontdict_labels)

        ax.set_xticks(ticks = range(40, 65, 5), labels = ["<45", "[45 - 49]", "[50 - 54]", "[55 - 59]", "[60 - 65]"])
        ax.set_xlabel("Age group (yrs.)", fontdict = fontdict_labels)

        ax.set_title("Blood Pressure Increases with Age in \nControls but not in Patients", fontdict = fontdict_title)
    

        handles = [
            Line2D([0], [0], color = 'white', marker = 'o', markerfacecolor = palette_sex['female'], markeredgecolor = 'black', label = 'Females'),
            Line2D([0], [0], color = 'white', marker = 'o', markerfacecolor = palette_sex['male'], markeredgecolor = 'black', label = 'Males'),
            Line2D([0], [0], color = "black", linestyle = "solid", label = "Patients"),
            Line2D([0], [0], color = "black", linestyle = "dotted", label = "Controls")
        ]

        ax.legend(handles = handles, title = None, edgecolor = "black")

        plt.tight_layout()
        st.pyplot(fig)

        st.write(f"""
        <div class = 'all'>
            <h3>Variable Analysis</h3>
            <p>
                Both systolic and diastolic blood pressure displayed <b>substantial variability</b>, with wide range covered; and
            strongly correlated one with the other ({pval_txt(spearmanr(df['ap_hi'], df['ap_lo'])[1])}). With both systolic and
            diastolic blood pressure, <b>higher values were measured among patients with cardiovascular disease</b>
            ({pval_txt(mwu_cardio(df, ap_hi))} for systolic,  {pval_txt(mwu_cardio(df, ap_lo))} for diastolic 
            and {pval_txt(mwu_cardio(df, ap_m))} for mean blood pressure). 
            Such observations were expected as high blood pressure is known to be a risk factor
            for cardiovascular diseases.
            </p>
            <p>
            Prevalence increased with American Heart Association's categories, in a linear manner between categories
            "Normal", "Elevated" and "Hypertension Stage 1". There was an increase of approximately
            <b>11%</b> between these groups. However, prevalence <b>almost doubled</b> between categories
            "Hypertension Stage I (<b>{cardio_aha[cardio_aha['ap_aha'] == "3"]['cardio_1'].values[0]:.1f}%</b>)
            and "Hypertension Stage II" (<b>{cardio_aha[cardio_aha['ap_aha'] == "4"]['cardio_1'].values[0]:.1f}%</b>).
            </p>
            <p>
            Overall, <b>males subjects displayed higher levels</b> for mean blood pressure, 
            regardless of age or CV status. This difference tended to disminish with increasing age in controls, while it
            remained consistent in patients. Additionnaly, the mean blood pressure value was not affected by age in patients,
            whereas it increased with age in controls.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.write(f"""
    <div class = 'all conclusion'>
        <h3>Conclusion</h3>
        <ul>
            <li>Only {df['ap_aha'].value_counts(normalize=True)['1']:.1%} of subjects had blood pressure classified as "Normal"</li>
            <li>Males displayed higher blood pressure values than females, regardless of cardiovascular status</li>
            <li>Blood pressure increased with age among controls, but remained almost constant in patients</li>
            <li>Prevalence of CV disease increased with AHA's categories, up to {cardio_aha[cardio_aha['ap_aha'] == "4"]['cardio_1'].values[0]:.1f}% among subjects
            in the "Hypertension Stage II" group</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    #############################################
    ######### GLUCOSE AND CHOLESTEROL ###########
    #############################################

    elif feature_selection == "Glucose and Cholesterol":
        st.write(f"""
        <div class = 'all'>
            <h1>Glucose and Cholesterol</h1>
            <h3>Variables Description</h3>
        </div>""", unsafe_allow_html=True)

        var = [cholesterol, gluc]
        table = f"""
        <table class = "table_1" style = "width: 75% !important">
            <tr class = "head_tr">
                <th>Parameter</th>
                <th>Modalities</th>
                <th>Observations (<i>n</i>)</th>
                <th>Observations (%)</th>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td rowspan = "3" style = 'font-weight: bold'>{var[0].full_name}</td>
                <td style = 'border-left: 1px dashed black'>{var[0].mod_names[0]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[0].name].value_counts()[var[0].mod[0]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[0].name].value_counts(normalize = True)[var[0].mod[0]]:.1%}</td>
            </tr>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td style = 'border-left: 1px dashed black'>{var[0].mod_names[1]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[0].name].value_counts()[var[0].mod[1]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[0].name].value_counts(normalize = True)[var[0].mod[1]]:.1%}</td>
            </tr>   
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td style = 'border-left: 1px dashed black'>{var[0].mod_names[2]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[0].name].value_counts()[var[0].mod[2]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[0].name].value_counts(normalize = True)[var[0].mod[2]]:.1%}</td>
            </tr>
            </tr>
            <tr style = 'border-top: 3px solid black'>
                <td rowspan = "3" style = 'font-weight: bold'>{var[1].full_name}</td>
                <td style = 'border-left: 1px dashed black'>{var[1].mod_names[0]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[1].name].value_counts()[var[1].mod[0]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[1].name].value_counts(normalize = True)[var[1].mod[0]]:.1%}</td>
            </tr>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td style = 'border-left: 1px dashed black'>{var[1].mod_names[1]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[1].name].value_counts()[var[1].mod[1]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[1].name].value_counts(normalize = True)[var[1].mod[1]]:.1%}</td>
            </tr>   
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td style = 'border-left: 1px dashed black'>{var[1].mod_names[2]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[1].name].value_counts()[var[1].mod[2]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[1].name].value_counts(normalize = True)[var[1].mod[2]]:.1%}</td>
            </tr>
        </table><br>"""
        
        st.write(table, unsafe_allow_html=True)    


        st.write('<h3>Data Visualisations</h3>', unsafe_allow_html=True)
        ##### FIGURE #####

        fig = plt.figure(figsize = (12, 12))
        plt.subplots_adjust(hspace = 0.6, wspace = 0.5)
        spec = gridspec.GridSpec(nrows = 4, ncols = 4)

        # Barplots Chol and Gluc

        df_gluc = df['gluc'].value_counts(normalize = True).reset_index(name = "glucose").rename(columns = {'index' : 'score'})
        df_chol = df['cholesterol'].value_counts(normalize = True).reset_index().rename(columns = {'index' : 'score'})
        df_gc_temp = df_gluc.merge(df_chol, how='inner', left_on='score', right_on='score')
        df_gc = pd.melt(df_gc_temp, id_vars = "score", var_name = "parameter", value_name = "percentage")
        
        palette_gc = ['palegreen', 'orange', 'orangered']

        ax = fig.add_subplot(spec[0:2, 0:2])
        sns.barplot(
            x = 'parameter', 
            y = 'percentage', 
            hue = 'score', 
            data = df_gc, 
            ax = ax, 
            edgecolor = "black", 
            palette = palette_gc)
    
        ax.set_ylim(0,1)    
        ax.set_yticks(ticks = [x/100 for x in range(0,105,10)], labels = range(0,105,10))
        ax.set_ylabel('Subjects (%)', fontdict = fontdict_labels)

        ax.set_xticklabels([tick.get_text().capitalize() for tick in ax.get_xticklabels()], size = 12)
        ax.set_xlabel(None)

        ax.set_title("Distribution of Glucose and \nCholesterol Levels", fontdict = fontdict_title)

        handles = [
            Patch(facecolor = "palegreen", label = "Normal", edgecolor = "black"),
            Patch(facecolor = "orange", label = "Above normal", edgecolor = "black"),
            Patch(facecolor = "orangered", label = "Well above normal", edgecolor = "black")
        ]

        ax.legend(handles = handles, title = None, edgecolor = "black", facecolor = 'white',framealpha=1, bbox_to_anchor = (0.75, 0.6))

        # PIEPLOT Glucose Males and Females

        pie_gluc_f = []
        pie_gluc_m = []
        pie_chol_f = []
        pie_chol_m = []

        for i in ["1", "2", "3"]:
            pie_gluc_f.append(df[df['sex'] == "female"]["gluc"].value_counts(normalize = True)[i])
            pie_gluc_m.append(df[df['sex'] == "male"]["gluc"].value_counts(normalize = True)[i])
            pie_chol_f.append(df[df['sex'] == "female"]["cholesterol"].value_counts(normalize = True)[i])
            pie_chol_m.append(df[df['sex'] == "male"]["cholesterol"].value_counts(normalize = True)[i])

        ax = fig.add_subplot(spec[0,2])

        ax.pie(
            x = pie_gluc_f,  
            colors = palette_gc, 
            autopct = '%1.1f%%',
            pctdistance = 1.3,
            startangle = 90, 
            wedgeprops = {'ec' : 'black'},
            textprops = {'weight' : 'bold'}
        )

        ax.text(1.8, 1.8, "Comparable Glucose Levels in Males and Females", fontdict = fontdict_title, ha = "center", va = "center")
        ax.text(-1, 1, "Females", fontstyle = 'italic', fontweight = 'bold', fontsize = 12, ha = "center", va = "center")


        ax = fig.add_subplot(spec[0,3])

        ax.pie(
            x = pie_gluc_m, 
            colors = palette_gc, 
            autopct = '%1.1f%%', 
            pctdistance = 1.3,
            startangle = 90, 
            wedgeprops = {'ec' : 'black'},
            textprops = {'weight' : 'bold'}
        )

        ax.text(-1, 1, "Males", fontstyle = 'italic', fontweight = 'bold', fontsize = 12, ha = "center", va = "center")


        ax = fig.add_subplot(spec[1,2])

        ax.pie(
            x = pie_chol_f,  
            colors = palette_gc, 
            autopct = '%1.1f%%',
            pctdistance = 1.4,
            startangle = 90, 
            wedgeprops = {'ec' : 'black'},
            textprops = {'weight' : 'bold'}
        )
        ax.text(1.8, 1.8, "Fewer Females had Normal \nCholesterol Levels", fontdict = fontdict_title, ha = "center", va = "center")
        ax.text(-1, 1, "Females", fontstyle = 'italic', fontweight = 'bold', fontsize = 12, ha = "center", va = "center")


        ax = fig.add_subplot(spec[1,3])

        ax.pie(
            x = pie_chol_m, 
            colors = palette_gc, 
            autopct = '%1.1f%%', 
            pctdistance = 1.4,
            startangle = 90, 
            wedgeprops = {'ec' : 'black'},
            textprops = {'weight' : 'bold'}
        )

        ax.text(-1, 1, "Males", fontstyle = 'italic', fontweight = 'bold', fontsize = 12, ha = "center", va = "center")


        # CV prevalence and Gluc

        ax = fig.add_subplot(spec[2:4, 0:2])

        df_gluc = df.groupby('gluc')['cardio'].value_counts(normalize = True).unstack().reset_index()
        df_gluc = df_gluc.rename(columns = {"gluc" : "glucose", "0" : "cardio_0", "1" : "cardio_1"})

        sns.lineplot(
            x = "glucose", 
            y = "cardio_1", 
            data = df_gluc,
            color = "darkcyan", 
            ax = ax, 
            linewidth = 3,
            marker = "o",
            markeredgecolor = 'black'
        )

        for i in [1, 2, 3]:
            prevalence = df_gluc[df_gluc['glucose'] == str(i)]["cardio_1"].values[0]
            ax.text(i-1, prevalence + 0.04, f'{prevalence:.1%}', fontweight = 'bold', ha = "right")

        ax.set_ylim(0,1)
        ax.set_yticks(ticks = [x/10 for x in range(0, 11, 1)], labels = range(0,110,10))
        ax.set_ylabel('CVD Prevalence (%)', fontdict = fontdict_labels)

        ax.set_xlim(-0.5, 2.3)
        ax.set_xticks(ax.get_xticks(), ['Normal', 'Above normal', 'Well above \nnormal'], size = 12)
        ax.set_xlabel("Glucose Level", fontdict = fontdict_labels)

        ax.set_title('Cardiovascular Diseases Prevalence \nIncreases with Glucose Levels', fontdict = fontdict_title)

        # CV prevalence and Chol

        ax = fig.add_subplot(spec[2:4, 2:4])

        df_chol = df.groupby('cholesterol')['cardio'].value_counts(normalize = True).unstack().reset_index()
        df_chol = df_chol.rename(columns = {"0" : "cardio_0", "1" : "cardio_1"})

        sns.lineplot(
            x = "cholesterol", 
            y = "cardio_1", 
            data = df_chol, 
            color = "gold",
            linewidth = 3,
            marker = "o",
            ax = ax, 
            markeredgecolor = 'black'
        )

        for i in [1, 2, 3]:
            prevalence = df_chol[df_chol['cholesterol'] == str(i)]["cardio_1"].values[0]
            ax.text(i-1, prevalence + 0.04, f'{prevalence:.1%}', fontweight = 'bold', ha = "right")

        ax.set_ylim(0,1)
        ax.set_yticks(ticks = [x/10 for x in range(0, 11, 1)], labels = range(0,110,10))
        ax.set_ylabel('CVD Prevalence (%)', fontdict = fontdict_labels)

        ax.set_xlim(-0.5, 2.3)
        ax.set_xticks(ax.get_xticks(), ['Normal', 'Above normal', 'Well above \nnormal'], size = 12)
        ax.set_xlabel("Cholesterol Level", fontdict = fontdict_labels)

        ax.set_title('Cardiovascular Diseases Prevalence \nIncreases with Cholesterol Levels', fontdict = fontdict_title)

        st.pyplot(fig)

        # Table: Chol & Gluc

        st.write(f"""
        <div class = 'all'>
            <h3>Cross Tabulation</h3>
        </div>
        """, unsafe_allow_html=True)
        tab_gc = pd.crosstab(df['cholesterol'], df['gluc'], normalize = 'all')

        st.write(f"""
        <div class = 'all'>
            <table style = "width : 80%">
                <tr style = 'background-color : white; border: none'>
                    <th style = 'border: none'></th>
                    <th style = 'border: none'></th>
                    <th colspan = '3' style = 'border: 1px solid black; background-color: darkcyan; color: white'>Glucose</th>
                </tr>
                <tr style = 'border: none'>
                    <th style = 'border: none'></th>
                    <th style = 'border: none'></th>
                    <td style = 'border: 1px solid black'>Normal</td>
                    <td style = 'border: 1px solid black'>Above normal</td>
                    <td style = 'border: 1px solid black'>Well above normal</td>
                </tr>
                <tr>
                    <th rowspan = "3" style = 'border: 1px solid black; background-color: gold'>Cholesterol</th>
                    <td style = 'border: 1px solid black'>Normal</td>
                    <td style = 'border: 1px solid black'>{tab_gc.loc["1", "1"]:.1%}</td>
                    <td style = 'border: 1px solid black'>{tab_gc.loc["1", "2"]:.1%}</td>
                    <td style = 'border: 1px solid black'>{tab_gc.loc["1", "3"]:.1%}</td>            
                </tr>
                <tr>
                    <td style = 'border: 1px solid black'>Above normal</td>
                    <td style = 'border: 1px solid black'>{tab_gc.loc["2", "1"]:.1%}</td>
                    <td style = 'border: 1px solid black'>{tab_gc.loc["2", "2"]:.1%}</td>
                    <td style = 'border: 1px solid black'>{tab_gc.loc["2", "3"]:.1%}</td>    
                </tr>
                <tr>
                    <td style = 'border: 1px solid black'>Well above normal</td>
                    <td style = 'border: 1px solid black'>{tab_gc.loc["3", "1"]:.1%}</td>
                    <td style = 'border: 1px solid black'>{tab_gc.loc["3", "2"]:.1%}</td>
                    <td style = 'border: 1px solid black'>{tab_gc.loc["3", "3"]:.1%}</td>                
                </tr>
            </table><br>
        </div>
        """, unsafe_allow_html=True)

        # Grid: Chol & Gluc

        st.write(f"""
        <div class = 'all'>
        <p style = 'color: navy; font-size: 18px; font-weight: bold'>Cardiovascular Status according to Glucose and Cholesterol</p>
        </div>
        """, unsafe_allow_html=True)

        df_percentage = df.groupby(['cholesterol', 'gluc', 'cardio']).size().reset_index(name='count')
        df_percentage['percentage'] = df_percentage.groupby(['cholesterol', 'gluc'])['count'].apply(lambda x: x / x.sum() * 100)

        g = sns.FacetGrid(df_percentage, col="gluc", row="cholesterol", margin_titles=True, row_order = ["3","2","1"])
        g.map_dataframe(sns.barplot, x="cardio", y="percentage", palette= palette_cardio, edgecolor = 'black')

        for ax in g.axes.flat:
            
            ax.set_ylim(0, 100)
            ax.set_yticks(ticks = range(0,110,10), labels = range(0,110,10))

            
            bar_y = [bar.get_height() for bar in ax.patches]
            
            ax.text(0, bar_y[0]+5, f"{bar_y[0]:.1f}", ha = "center", fontsize=10, fontweight='bold')
            ax.text(1, bar_y[1]+5, f"{bar_y[1]:.1f}", ha = "center", fontsize=10, fontweight='bold')

        st.pyplot(g)

        pval_chi2_gc = chi2_var(df, gluc, cholesterol)
        pval_chi2_gluc = chi2_cardio(df, gluc)
        pval_chi2_chol = chi2_cardio(df, cholesterol)


        st.write(f"""
        <div class = 'all'>
            <h3>Variable Analysis</h3>
            <p>
                A large majority of subjects in the cohort had normal values for either glucose 
                (<b>{df["gluc"].value_counts(normalize = True)["1"]:.1%}</b>) or cholesterol
                (<b>{df["cholesterol"].value_counts(normalize = True)["1"]:.1%}</b>). Proportions of subjects
                labeled as "Above normal" and "Well above normal" were comparable in both variables.
            </p>
            <p>
                The prevalence of cardiovascular disease <b>increased</b> with the levels of both glucose and cholesterol. Prevalence 
                was of <b>{df[df["gluc"] == "1"]['cardio'].value_counts(normalize = True)["1"]:.1%}</b> among subjects with normal glucose,
                and of <b>{df[df["cholesterol"] == "1"]['cardio'].value_counts(normalize = True)["1"]:.1%}</b> among subjects with
                normal cholesterol. Prevalence <b>linearly</b> correlated with increasing levels of cholesterol, up to 
                <b>{df[df["cholesterol"] == "3"]['cardio'].value_counts(normalize = True)["1"]:.1%}</b> among subjects with cholesterol
                labeled as "Well above normal". Prevalence also increased with glucose levels, but remained constant between subjects
                with glucose labeled as "Above normal" and "Well above normal" (around 
                <b>{np.mean([df[df["gluc"] == "2"]['cardio'].value_counts(normalize = True)["1"], 
                df[df["gluc"] == "3"]['cardio'].value_counts(normalize = True)["1"]]):.0%}</b>).        
            </p>
            <p>
                Analysing both glucose and cholesterol levels together revealed that large majority of subjects had normal levels
                of both parameters (<b>{tab_gc.loc["1", "1"]:.1%}</b>). Interestingly, prevalence of CV disease remained relatively constant
                with increasing glucose levels, for a fixed cholesterol level. This suggest that while both parameters correlated 
                with cardiovascular disease status ({pval_chi2_gluc} for glucose and {pval_chi2_chol} for cholesterol),
                the feature <code>gluc</code> may bring limited information due to the influence of <code>cholesterol</code>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.write(f"""
        <div class = 'all conclusion'>
            <h3>Conclusion</h3>
            <ul>
                <li>Majority of subjects had normal values for both glucose and cholesterol ({tab_gc.loc["1", "1"]:.1%})</li>
                <li>Prevalence of cardiovascular diseases increased with glucose and cholesterol</li>
                <li>Modalities "Above normal" and "Well above normal" could be merged for <code>gluc</code></li>
                <li>Dropping <code>gluc</code> may be considered as it may bring a limited amount of information compared to 
                <code>cholesterol</code></li>
                <li>Cut-off values for both <code>gluc</code> and <code>cholesterol</code> modalities are not known, there could
                be some bias, they therefore need to be interpreted with caution</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    #####################
    ##### LIFESTYLE #####
    #####################
        
    elif feature_selection == "Lifestyle":
        st.write(f"""
            <div class = 'all'>
                <h1>Lifestyle</h1>
                <h3>Variables Description</h3>
            </div>
            """, unsafe_allow_html=True)

        var = [smoke, alco, active, lifestyle, healthy_ls]
        table = f"""
        <table class = "table_1" style = "width: 75% !important">
            <tr class = "head_tr">
                <th>Parameter</th>
                <th>Modalities</th>
                <th>Observations (<i>n</i>)</th>
                <th>Observations (%)</th>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td rowspan = "2" style = 'font-weight: bold'>{var[0].full_name}</td>
                <td style = 'border-left: 1px dashed black'>{(var[0]).mod_names[0]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[0].name].value_counts()[var[0].mod[0]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[0].name].value_counts(normalize = True)[var[0].mod[0]]:.1%}</td>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td style = 'border-left: 1px dashed black'>{var[0].mod_names[1]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[0].name].value_counts()[var[0].mod[1]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[0].name].value_counts(normalize = True)[var[0].mod[1]]:.1%}</td>
            </tr>   
            <tr style = 'border-top: 3px solid black'>
                <td rowspan = "2" style = 'font-weight: bold'>{var[1].full_name}</td>
                <td style = 'border-left: 1px dashed black'>{var[1].mod_names[0]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[1].name].value_counts()[var[1].mod[0]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[1].name].value_counts(normalize = True)[var[1].mod[0]]:.1%}</td>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td style = 'border-left: 1px dashed black'>{var[1].mod_names[1]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[1].name].value_counts()[var[1].mod[1]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[1].name].value_counts(normalize = True)[var[1].mod[1]]:.1%}</td>
            </tr> 
            <tr style = 'border-top: 3px solid black'>
                <td rowspan = "2" style = 'font-weight: bold'>{var[2].full_name}</td>
                <td style = 'border-left: 1px dashed black'>{(var[2]).mod_names[0]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[2].name].value_counts()[var[2].mod[0]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[2].name].value_counts(normalize = True)[var[2].mod[0]]:.1%}</td>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td style = 'border-left: 1px dashed black'>{var[2].mod_names[1]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[2].name].value_counts()[var[2].mod[1]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[2].name].value_counts(normalize = True)[var[2].mod[1]]:.1%}</td>
            </tr>  
            <tr style = 'border-top: 3px solid black'>
                <td rowspan = "8" style = 'font-weight: bold'>{var[3].full_name}</td>
                <td style = 'border-left: 1px dashed black'>{(var[3]).mod_names[0]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[3].name].value_counts()[var[3].mod[0]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[3].name].value_counts(normalize = True)[var[3].mod[0]]:.1%}</td>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td style = 'border-left: 1px dashed black'>{var[3].mod_names[1]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[3].name].value_counts()[var[3].mod[1]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[3].name].value_counts(normalize = True)[var[3].mod[1]]:.1%}</td>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td style = 'border-left: 1px dashed black'>{var[3].mod_names[2]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[3].name].value_counts()[var[3].mod[2]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[3].name].value_counts(normalize = True)[var[3].mod[2]]:.1%}</td>
            </tr>  
            <tr style = 'border-top: 1px solid black'>
                <td style = 'border-left: 1px dashed black'>{var[3].mod_names[3]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[3].name].value_counts()[var[3].mod[3]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[3].name].value_counts(normalize = True)[var[3].mod[3]]:.1%}</td>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td style = 'border-left: 1px dashed black'>{var[3].mod_names[4]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[3].name].value_counts()[var[3].mod[4]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[3].name].value_counts(normalize = True)[var[3].mod[4]]:.1%}</td>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td style = 'border-left: 1px dashed black'>{var[3].mod_names[5]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[3].name].value_counts()[var[3].mod[5]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[3].name].value_counts(normalize = True)[var[3].mod[5]]:.1%}</td>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td style = 'border-left: 1px dashed black'>{var[3].mod_names[6]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[3].name].value_counts()[var[3].mod[6]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[3].name].value_counts(normalize = True)[var[3].mod[6]]:.1%}</td>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td style = 'border-left: 1px dashed black'>{var[3].mod_names[7]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[3].name].value_counts()[var[3].mod[7]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[3].name].value_counts(normalize = True)[var[3].mod[7]]:.1%}</td>
            </tr>
            <tr style = 'border-top: 3px solid black'>
                <td rowspan = "2" style = 'font-weight: bold'>{var[4].full_name}</td>
                <td style = 'border-left: 1px dashed black'>{(var[4]).mod_names[0]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[4].name].value_counts()[var[4].mod[0]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[4].name].value_counts(normalize = True)[var[4].mod[0]]:.1%}</td>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td style = 'border-left: 1px dashed black;'>{var[4].mod_names[1]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[4].name].value_counts()[var[4].mod[1]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var[4].name].value_counts(normalize = True)[var[4].mod[1]]:.1%}</td>
            </tr>
        </table><br>"""
        
        st.write(table, unsafe_allow_html=True)    

        st.write('<h3>Data Visualisations</h3>', unsafe_allow_html=True)

        ##### FIGURE #####

        fig = plt.figure(figsize = (12, 15))
        spec = gridspec.GridSpec(nrows = 3, ncols = 2)
        plt.subplots_adjust(hspace = 0.3)

        # boxplot Tobbaco

        df_smoke = pd.melt(
            df[['ap_hi', 'ap_lo', 'ap_m', 'smoke', 'cardio']], 
            id_vars = ['smoke', 'cardio'], 
            var_name = 'Pressure_type', 
            value_name = 'Pressure_value'
        )

        ax = fig.add_subplot(spec[0,0])
        sns.boxplot(
            x = 'Pressure_type', 
            y = 'Pressure_value', 
            hue = 'smoke', 
            data = df_smoke, 
            width = 0.5, 
            palette = ['cyan', 'gray'],
            medianprops = {'color' : 'black'}, 
            boxprops = {'edgecolor' : 'black'},
            capprops = {'color' : 'black'},
            whiskerprops = {'color' : 'black'},
            flierprops = {'color' : 'black'},
            ax = ax
        )

        median_sys_smoke_0 = np.median(df[df['smoke'] == "0"]["ap_hi"])
        median_sys_smoke_1 = np.median(df[df['smoke'] == "1"]["ap_hi"])

        ax.text(-0.47, median_sys_smoke_0, f"{median_sys_smoke_0:.0f}", fontweight = 'bold', ha = 'left', va = "center")
        ax.text(0.47, median_sys_smoke_1, f"{median_sys_smoke_1:.0f}", fontweight = 'bold', ha = 'right', va = "center")

        median_dia_smoke_0 = np.median(df[df['smoke'] == "0"]["ap_lo"])
        median_dia_smoke_1 = np.median(df[df['smoke'] == "1"]["ap_lo"])

        ax.text(1-0.43, median_dia_smoke_0, f"{median_dia_smoke_0:.0f}", fontweight = 'bold', ha = 'left', va = "center")
        ax.text(1+0.43, median_dia_smoke_1, f"{median_dia_smoke_1:.0f}", fontweight = 'bold', ha = 'right', va = "center")

        median_m_smoke_0 = np.median(df[df['smoke'] == "0"]["ap_m"])
        median_m_smoke_1 = np.median(df[df['smoke'] == "1"]["ap_m"])

        ax.text(2-0.43, median_m_smoke_0, f"{median_m_smoke_0:.0f}", fontweight = 'bold', ha = 'left', va = "center")
        ax.text(2+0.43, median_m_smoke_1, f"{median_m_smoke_1:.0f}", fontweight = 'bold', ha = 'right', va = "center")



        ax.set_ylim(0,250)
        ax.set_yticks(ticks = range(0, 275, 25))
        ax.set_ylabel("Blood Pressure (mmHg)", fontdict = fontdict_labels)

        ax.set_xticks(ticks = ax.get_xticks(), labels = ['Systolic', 'Diastolic', 'Mean'])
        ax.set_xlabel('Blood Pressure Type', fontdict = fontdict_labels)

        ax.set_title('Blood Pressure was not Higher \nin Smokers', fontdict = fontdict_title)

        handles = [
            Patch(facecolor = 'cyan', edgecolor = 'black', label = 'Non-smoker'),
            Patch(facecolor = 'gray', edgecolor = 'black', label = 'Smoker'),
        ]

        ax.legend(handles = handles, title = None, edgecolor = 'black')

        # boxplot Active

        df_active = pd.melt(
            df[['ap_hi', 'ap_lo', 'ap_m', 'active', 'cardio']], 
            id_vars = ['active', 'cardio'], 
            var_name = 'Pressure_type', 
            value_name = 'Pressure_value'
        )

        ax = fig.add_subplot(spec[0,1])
        sns.boxplot(
            x = 'Pressure_type', 
            y = 'Pressure_value', 
            hue = 'active', 
            data = df_active, 
            width = 0.5, 
            palette = ['gray', 'chartreuse'],
            medianprops = {'color' : 'black'}, 
            boxprops = {'edgecolor' : 'black'},
            capprops = {'color' : 'black'},
            whiskerprops = {'color' : 'black'},
            flierprops = {'color' : 'black'},
            ax = ax
        )

        median_sys_active_0 = np.median(df[df['active'] == "0"]["ap_hi"])
        median_sys_active_1 = np.median(df[df['active'] == "1"]["ap_hi"])

        ax.text(-0.47, median_sys_active_0, f"{median_sys_active_0:.0f}", fontweight = 'bold', ha = 'left', va = "center")
        ax.text(0.47, median_sys_active_1, f"{median_sys_active_1:.0f}", fontweight = 'bold', ha = 'right', va = "center")

        median_dia_active_0 = np.median(df[df['active'] == "0"]["ap_lo"])
        median_dia_active_1 = np.median(df[df['active'] == "1"]["ap_lo"])

        ax.text(1-0.43, median_dia_active_0, f"{median_dia_active_0:.0f}", fontweight = 'bold', ha = 'left', va = "center")
        ax.text(1+0.43, median_dia_active_1, f"{median_dia_active_1:.0f}", fontweight = 'bold', ha = 'right', va = "center")

        median_m_active_0 = np.median(df[df['active'] == "0"]["ap_m"])
        median_m_active_1 = np.median(df[df['active'] == "1"]["ap_m"])

        ax.text(2-0.43, median_m_active_0, f"{median_m_active_0:.0f}", fontweight = 'bold', ha = 'left', va = "center")
        ax.text(2+0.43, median_m_active_1, f"{median_m_active_1:.0f}", fontweight = 'bold', ha = 'right', va = "center")

        ax.set_ylim(0,250)
        ax.set_yticks(ticks = range(0, 275, 25))
        ax.set_ylabel("Blood Pressure (mmHg)", fontdict = fontdict_labels)

        ax.set_xticks(ticks = ax.get_xticks(), labels = ['Systolic', 'Diastolic', 'Mean'])
        ax.set_xlabel('Blood Pressure Type', fontdict = fontdict_labels)

        ax.set_title('Physical Activity did not Influence \n Blood Pressure', fontdict = fontdict_title)

        handles = [
            Patch(facecolor = 'gray', edgecolor = 'black', label = 'Not active'),
            Patch(facecolor = 'chartreuse', edgecolor = 'black', label = 'Active'),
            Text("120", color='black', ha='center', va='center', fontweight='bold')
        ]

        ax.legend(handles = handles, title = None, edgecolor = 'black')

        # Lifestyle barplot

        ax = fig.add_subplot(spec[1,:])


        df_lifestyle = (df.groupby('lifestyle')['cardio'].value_counts(normalize = True)*100).sort_index().unstack().reset_index().rename(columns = {"0": "cardio_0", "1" : "cardio_1"})
        sns.barplot(
            x = "lifestyle",
            y = "cardio_1", 
            data = df_lifestyle, 
            edgecolor = "black", 
            ax = ax,
            palette = ["cornsilk", "yellow", "salmon", "lightskyblue", "orangered", "greenyellow", "blueviolet", "darkslategray"]
        )
        ax.axhline(y = 50, linestyle = "dashed", color = "black")



        ax.set_ylim(0,100)
        ax.set_yticks(range(0,110,10))
        ax.set_ylabel("Prevalence (%)", fontdict = fontdict_labels)

        ax.set_xticks(ticks = ax.get_xticks(),
                    labels = [
            "Non-smoker\nNo alcohol\nActive",
            "Smoker",
            "Alcohol",
            "Not Active",
            "Smoker\nAlcohol",
            "Smoker\nNot active",
            "Alcohol\nNot active",
            "Smoker\nAlcohol\nNot active"    
        ], 
                    size = 12)

        ax.set_xlabel(None)

        ax.set_title("No Strong Influence of Lifestyle was found on Cardiovascular Diseases Prevalence", fontdict = fontdict_title)

        # barplot sex ratio lifestyle

        ax = fig.add_subplot(spec[2, :])

        df_sex_active = (df.groupby('sex')['active'].value_counts(normalize = True)*100).sort_index().unstack().reset_index().rename(columns = {"0": "active_0", "1" : "active_1"})
        df_sex_smoke = (df.groupby('sex')['smoke'].value_counts(normalize = True)*100).sort_index().unstack().reset_index().rename(columns = {"0": "smoke_0", "1" : "smoke_1"})
        df_sex_alco = (df.groupby('sex')['alco'].value_counts(normalize = True)*100).sort_index().unstack().reset_index().rename(columns = {"0": "alco_0", "1" : "alco_1"})
        df_sex_healthy = (df.groupby('sex')['healthy_ls'].value_counts(normalize = True)*100).sort_index().unstack().reset_index().rename(columns = {"0": "healthy_0", "1" : "healthy_1"})
        df_sex_ls = df_sex_active.merge(df_sex_smoke, on = "sex", how = "inner").merge(df_sex_alco, on = "sex", how = "inner").merge(df_sex_healthy, on = "sex", how = "inner")
        df_sex_ls = pd.melt(df_sex_ls.drop(['active_0', 'smoke_0', 'alco_0', 'healthy_0'], axis = 1), id_vars = "sex", var_name = "parameter", value_name = "percentage")

        sns.barplot(
            x = "parameter",
            y = "percentage",
            hue = "sex", 
            data = df_sex_ls,
            palette = palette_sex, 
            ax = ax,
            edgecolor = 'black',
            order = ['smoke_1', 'alco_1', 'active_1', 'healthy_1']
        )

        for container in ax.containers:
            for bar in container:
                y = bar.get_height()
                x = bar.get_x()
                ax.text(x + 0.2, y + 5, f'{y:.1f}%', fontweight='bold', fontsize = 12, ha = "center")


        ax.set_ylim(0,100)
        ax.set_yticks(ticks = range(0,110,10), labels = range(0,110,10))
        ax.set_ylabel('Percentage (%)', fontdict = fontdict_labels)

        ax.set_xticks(ticks = ax.get_xticks(), labels = ['Smoker', 'Alcohol', 'Active','Healthy\nlifestyle'], size = 12)
        ax.set_xlabel(None)

        ax.set_title("Females were less Likely to Smoke and Drink Alcohol", fontdict = fontdict_title)

        ax.legend(title = None, edgecolor = 'black')

        st.pyplot(fig)

        st.write(f"""
        <div class = 'all'>
            <h3>Variables analysis</h3>
            <p>
                A minority of subjects reported smoking (<b>{df['smoke'].value_counts(normalize = True)["1"]:.1%}</b>) or alcohol consumption
                (<b>{df['alco'].value_counts(normalize = True)["1"]:.1%}</b>), whereas they were <b>{df['active'].value_counts(normalize = True)["1"]:.1%}</b>
                to report some physical activity. This cohort therefore displayed overall healthy lifestyle, as 
                <b>{df['healthy_ls'].value_counts(normalize = True)["1"]:.1%}</b> of subject were non-smoker, did not drink alcohol, and had some 
                physical activity. On the other hand, there were only <b>{df['lifestyle'].value_counts(normalize = True)["7"]:.1%}</b>
                to report smoking, drinking alcohol and not exercising. Nevertheless, it should be kept in mind that the definition of 
                alcohol consumption and physical activity are not specified, and could therefore be biased.
            </p>
            <p>
                Surprisingly, <b>no obvious effect</b> of either either smoking or activity on blood pressure was found: median values were
                identical for systolic, diastolic and mean blood pressure, and distributions were comparable.        
            </p>
            <p>
                Finally, male subjects were <b>more likely</b> to smoke and drink alcohol than females: 
                <b>{df_sex_ls[(df_sex_ls['sex'] == 'male') & (df_sex_ls['parameter'] == "smoke_1")]['percentage'].values[0]:.1f}%</b> of males were
                smokers and <b>{df_sex_ls[(df_sex_ls['sex'] == 'male') & (df_sex_ls['parameter'] == "alco_1")]['percentage'].values[0]:.1f}%</b> drank
                alcohol, compared to <b>{df_sex_ls[(df_sex_ls['sex'] == 'female') & (df_sex_ls['parameter'] == "smoke_1")]['percentage'].values[0]:.1f}%</b>
                and <b>{df_sex_ls[(df_sex_ls['sex'] == 'female') & (df_sex_ls['parameter'] == "alco_1")]['percentage'].values[0]:.1f}%</b> respectively for females.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.write(f"""
        <div class = 'all conclusion'>
            <h3>Conclusion</h3>
            <ul>
                <li>A small number of subjects reported smoking and or drinking alcohol, and these variables did not 
                display any obvious correlation with <code>cardio</code></li>
                <li>Smoking or exercising did not influence blood pressure levels</li>
                <li>A tendency of higher CV disease prevalence could exist among subjects that do not exercise</li>
                <li>Males were more likely to smoke and drink alcohol, while a comparable proportion of males and females
                reported physical activity</li>
                <li><code>smoke</code> should not be included in the model due to plausible bias</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        ##########################
        ##### TARGET FEATURE #####
        ##########################

    elif feature_selection == "Target Feature":
        st.write(f"""
        <div class = 'all'>
            <h1>Target Variable</h1>
            <h3>Variable Description</h3>
        </div>""", unsafe_allow_html=True)

        var = cardio

        table = f"""
        <table class = "table_1" style = "width: 75% !important">
            <tr class = "head_tr">
                <th>Parameter</th>
                <th>Modalities</th>
                <th>Observations (<i>n</i>)</th>
                <th>Observations (%)</th>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td rowspan = "2" style = 'font-weight: bold'>{var.full_name}</td>
                <td style = 'border-left: 1px dashed black'>{var.mod_names[0]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var.name].value_counts()[var.mod[0]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var.name].value_counts(normalize = True)[var.mod[0]]:.1%}</td>
            </tr>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td style = 'border-left: 1px dashed black'>{var.mod_names[1]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var.name].value_counts()[var.mod[1]]}</td>
                <td style = 'border-left: 1px dashed black'>{df[var.name].value_counts(normalize = True)[var.mod[1]]:.1%}</td>
            </tr>
        </table><br>"""
        
        st.write(table, unsafe_allow_html=True)    

        st.write(f"""
        <div class = 'all'>
            <h3>Subjects Characteristics</h3>
            <p>The target variable was completely balanced among observations, with half individuals suffering from
            cardiovascular disease. Characteristics of the cohort are summarised in the following table:
            </p>
            <table class = 'table_1'>
                <tr class = 'head_tr'>
                    <th>Variable</th>
                    <th>Patients <br><i>n</n> = {len(df[df['cardio'] == "1"])}</th>
                    <th>Controls <br><i>n</n> = {len(df[df['cardio'] == "0"])}</th>
                    <th>Cohort <br><i>n</n> = {len(df)}</th>
                </tr>
                <tr>
                    <td colspan = '4' style = 'background-color: lightcyan; font-variant: small-caps; text-align: left !important; line-height: 0.6; border: 3px solid black; font-weight: bold'>Demographic Variables</td>
                </tr>
                <tr>
                    <td>Age (years)</td>
                    <td>{mean_sd_1(df, age)}</td>
                    <td>{mean_sd_0(df, age)}</td>
                    <td>{mean_sd(df, age)}</td>
                </tr>
                <tr>
                    <td>Male Sex</td>
                    <td>{pp_1(df, sex)}</td>
                    <td>{pp_0(df, sex)}</td>
                    <td>{pp(df, sex)}</td>
                </tr>
                <tr>
                    <td>Height (cm)</td>
                    <td>{mean_sd_1(df, height)}</td>
                    <td>{mean_sd_0(df, height)}</td>
                    <td>{mean_sd(df, height)}</td>
                </tr>
                <tr>
                    <td>Weight (kg)</td>
                    <td>{mean_sd_1(df, weight)}</td>
                    <td>{mean_sd_0(df, weight)}</td>
                    <td>{mean_sd(df, weight)}</td>
                </tr>
                <tr>
                    <td>BMI (kg/m²)</td>
                    <td>{mean_sd_1(df, bmi)}</td>
                    <td>{mean_sd_0(df, bmi)}</td>
                    <td>{mean_sd(df, bmi)}</td>
                </tr>
                <tr>
                    <td colspan = '4' style = 'background-color: lightcyan; font-variant: small-caps; text-align: left !important; line-height: 0.6; border: 3px solid black; font-weight: bold'>Blood Pressure</td>
                </tr>
                <tr>
                    <td>Systolic Blood Pressure (mmHg)</td>
                    <td>{mean_sd_1(df, ap_hi)}</td>
                    <td>{mean_sd_0(df, ap_hi)}</td>
                    <td>{mean_sd(df, ap_hi)}</td>
                </tr>
                <tr>
                    <td>Diastolic Blood Pressure (mmHg)</td>
                    <td>{mean_sd_1(df, ap_lo)}</td>
                    <td>{mean_sd_0(df, ap_lo)}</td>
                    <td>{mean_sd(df, ap_lo)}</td>
                </tr>
                <tr>
                    <td>Mean Blood Pressure (mmHg)</td>
                    <td>{mean_sd_1(df, ap_m)}</td>
                    <td>{mean_sd_0(df, ap_m)}</td>
                    <td>{mean_sd(df, ap_m)}</td>
                </tr>
                <tr>
                    <td>Blood Pressure Classification (AHA)</td>
                    <td>{pp_4_cardio(df, ap_aha, "1")}</td>
                    <td>{pp_4_cardio(df, ap_aha, "0")}</td>
                    <td>{pp_4(df, ap_aha)}</td>
                </tr>
                <tr>
                    <td colspan = '4' style = 'background-color: lightcyan; font-variant: small-caps; text-align: left !important; line-height: 0.6; border: 3px solid black; font-weight: bold'>Laboratory Data</td>
                </tr>
                <tr>
                    <td>Cholesterol</td>
                    <td>{pp_3_cardio(df, cholesterol, "1")}</td>
                    <td>{pp_3_cardio(df, cholesterol, "0")}</td>
                    <td>{pp_3(df, cholesterol)}</td>
                </tr>  
                <tr>
                    <td>Glucose</td>
                    <td>{pp_3_cardio(df, gluc, "1")}</td>
                    <td>{pp_3_cardio(df, gluc, "0")}</td>
                    <td>{pp_3(df, gluc)}</td>
                </tr>
                <tr>
                    <td colspan = '4' style = 'background-color: lightcyan; font-variant: small-caps; text-align: left !important; line-height: 0.6; border: 3px solid black; font-weight: bold'>Lifestyle</td>
                </tr>
                <tr>
                    <td>Smoke</td>
                    <td>{pp_1(df, smoke)}</td>
                    <td>{pp_0(df, smoke)}</td>
                    <td>{pp(df, smoke)}</td>
                </tr>
                <tr>
                    <td>Alcohol</td>
                    <td>{pp_1(df, alco)}</td>
                    <td>{pp_0(df, alco)}</td>
                    <td>{pp(df, alco)}</td>
                </tr>
                <tr>
                    <td>Physical activity</td>
                    <td>{pp_1(df, active)}</td>
                    <td>{pp_0(df, active)}</td>
                    <td>{pp(df, active)}</td>
                </tr>
            </table>
            </p>
        </div>
        """, unsafe_allow_html=True)