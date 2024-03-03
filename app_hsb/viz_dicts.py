# Creating a class "Parameter" to access parameter-related information
from utils import pval_shapiro, pval_txt, mean_sd_range, chi2_cardio, chi2_var, mwu_cardio, mean_sd, mean_sd_1, mean_sd_0, pp, pp_0, pp_1, pp_3, pp_3_cardio, pp_4, pp_4_cardio
import pandas as pd
import numpy as np
from utils import load_data, load_raw_data
from scipy.stats import shapiro, mannwhitneyu, chi2_contingency, spearmanr


df = load_data()
df_raw = load_raw_data()

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

h_lo = np.percentile(df['height'], 2.5)
h_hi = np.percentile(df['height'], 97.5)
w_lo = np.percentile(df['weight'], 2.5)
w_hi = np.percentile(df['weight'], 97.5)

cardio_aha = (df.groupby('ap_aha')['cardio'].value_counts(normalize = True)*100).unstack().reset_index().rename(columns = {"0" : "cardio_0", "1" : "cardio_1"})

tab_gc = pd.crosstab(df['cholesterol'], df['gluc'], normalize = 'all')
pval_chi2_gc = chi2_var(df, gluc, cholesterol)
pval_chi2_gluc = chi2_cardio(df, gluc)
pval_chi2_chol = chi2_cardio(df, cholesterol)

df_sex_active = (df.groupby('sex')['active'].value_counts(normalize = True)*100).sort_index().unstack().reset_index().rename(columns = {"0": "active_0", "1" : "active_1"})
df_sex_smoke = (df.groupby('sex')['smoke'].value_counts(normalize = True)*100).sort_index().unstack().reset_index().rename(columns = {"0": "smoke_0", "1" : "smoke_1"})
df_sex_alco = (df.groupby('sex')['alco'].value_counts(normalize = True)*100).sort_index().unstack().reset_index().rename(columns = {"0": "alco_0", "1" : "alco_1"})
df_sex_healthy = (df.groupby('sex')['healthy_ls'].value_counts(normalize = True)*100).sort_index().unstack().reset_index().rename(columns = {"0": "healthy_0", "1" : "healthy_1"})
df_sex_ls = df_sex_active.merge(df_sex_smoke, on = "sex", how = "inner").merge(df_sex_alco, on = "sex", how = "inner").merge(df_sex_healthy, on = "sex", how = "inner")
df_sex_ls = pd.melt(df_sex_ls.drop(['active_0', 'smoke_0', 'alco_0', 'healthy_0'], axis = 1), id_vars = "sex", var_name = "parameter", value_name = "percentage")

viz_data = {
    "age" : {
        "title": "Age",
        "table": f"""<table>
                <tr>
                    <th>Parameter</th>
                    <th>Mean ± SD</th>
                    <th>Range</th>
                    <th>Normality</th>
                </tr>
                <tr>
                    <td class = "param-name">{age.full_name} ({age.unit})</td>
                    <td>{mean_sd_range(df, age)[0]}</td>
                    <td>{mean_sd_range(df, age)[1]}</td>
                    <td>{pval_shapiro(df, age)}</td>
                </tr>
            </table><br>""",
        "img": "age_fig.png",
        "analysis": f"""<p>
                The variable <b>{age.name}</b> is not normally distributed and shows a left skew. The minimal age value found
                in the original dataset was {np.min(df_raw['age'])/365.25:.1f} yo. There was {len(df_raw[df_raw['age']/365.25 < 32])}
                individuals below 32 years old, and there no subject in the age group [32 - 38] yo. None 
                of the subjects below 32 years old had cardiovascular disease, and they were considered outliers due to the existing age gap. It was decided to
                take them out of the analysis. Consequently, the range for <b>{age.name}</b> is
                <b>[{np.min(df[age.name]):.0f} - {np.max(df[age.name]):.0f}]</b> years old.
                The max age value is surprisingly low as cardiovascular diseases are common among older individuals,
                the reason why no data from elderly patients were recorded is not known, but was probably a protocol requirement.
            </p>
            <p>
                The prevalence of CV diseases is known to increase with age, as it is the case in this cohort. The proportion of
                patients in the age group [51 - 55] is <b>around 50%</b>. It is below 50% in younger individuals and above in older individuals.
                The mean age in patients is <b>{mean_sd_range(df[df['cardio'] == "1"], age)[0]}</b> years old compared to
                <b>{mean_sd_range(df[df['cardio'] == "0"], age)[0]}</b> years old in controls.
            </p>""",
        "conclusion": f"""<ul>
                <li>The mean age in the cohort was {mean_sd_range(df, age)[0]} yo.</li>
                <li>Age range was {mean_sd_range(df, age)[1]} yo: no children and no elderly people were included</li>
                <li>Patients in the cohort were older than controls, and the prevalence of CV disease linearly increased with age</li>
                <li>It could be worth adjusting the model on age, <i>ie</i> separating the data in two distinct dataframes with a cut-off value for age nearby 50 yo.</li>
            </ul>""",
    },
    "sex" : {
        "title" : "Sex",
        "table" : f"""<table>
            <tr>
                <th>Parameter</th>
                <th>Modalities</th>
                <th>Observations (<i>n</i>)</th>
                <th>Observations (%)</th>
            </tr>
            <tr>
                <td rowspan = "2" class = "param-name">{sex.full_name}</td>
                <td>{sex.mod[0]}</td>
                <td>{df[sex.name].value_counts()[sex.mod[0]]}</td>
                <td>{df[sex.name].value_counts(normalize = True)[sex.mod[0]]:.1%}</td>
            </tr>
            </tr>
            <tr>
                <td>{sex.mod[1]}</td>
                <td>{df[sex.name].value_counts()[sex.mod[1]]}</td>
                <td>{df[sex.name].value_counts(normalize = True)[sex.mod[1]]:.1%}</td>
            </tr>
        </table><br>""",
        "img" : "sex_fig.png",
        "analysis" : f"""<p>
                <b>sex</b> showed an important degree of imbalance, with roughly <b><sup>2</sup>/<sub>3</sub>
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
            </p>""",
        "conclusion": f"""<ul>
                <li>No obvious influence of sex was observed, as CV disease prevalence was very close to 50% in both groups</li>
                <li>Prevalence of CV diseases increased with age in a comparable manner in both males and females</li>
                <li>As there is almost twice as much females in the cohort, we may consider rebalancing the training data to a 1:1 sex
                ratio or train distinct models for males and females</li>
            </ul>"""
    },
    "bmi" : {
        "title" : "Body-Mass Index, Height and Weight",
        "table" : f"""<table>
                <tr>
                    <th>Parameter</th>
                    <th>Mean ± SD</th>
                    <th>Range</th>
                    <th>Normality</th>
                </tr>
                <tr>
                    <td class = "param-name">{bmi.full_name} ({bmi.unit})</td>
                    <td>{mean_sd_range(df, bmi)[0]}</td>
                    <td>{mean_sd_range(df, bmi)[1]}</td>
                    <td>{pval_shapiro(df, bmi)}</td>
                </tr>
                <tr>
                    <td class = "param-name">{weight.full_name} ({weight.unit})</td>
                    <td>{mean_sd_range(df, weight)[0]}</td>
                    <td>{mean_sd_range(df, weight)[1]}</td>
                    <td>{pval_shapiro(df, weight)}</td>
                </tr>
                <tr>
                    <td class = "param-name">{height.full_name} ({height.unit})</td>
                    <td>{mean_sd_range(df, height)[0]}</td>
                    <td>{mean_sd_range(df, height)[1]}</td>
                    <td>{pval_shapiro(df, height)}</td>
                </tr>
            </table><br>""",
        "img": "bmi_height_weight_fig.png",
        "analysis": f"""<p>
                A wide range of values was covered by both height ({mean_sd_range(df, height)[1]} cm) and weight
            ({mean_sd_range(df, weight)[1]} kg). Consequently, range for bmi was <b>{mean_sd_range(df, bmi)[1]}</b> kg/m².
            The median values for bmi was <b>{np.median(df['bmi']):.1f}</b> kg/m². Men were taller than women, 
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
            </p>""",
        "conclusion": f"""<ul>
                <li>Cardiovascular diseases prevalence increases with BMI</li>
                <li>Distribution of bmi is right-skewed, with a max value of {np.max(df['bmi']):.1f} kg/m², but outlying
                values may have a limited impact</li>
                <li>Among subjects with outlying values for height and/or weight, weight seemed to have
                a greater influence than height</li>
                <li>It may be relevant to assess the influence of extreme values for height and weight on the 
                model</li>
            </ul>"""
    },
    "bp": {
        "title" : "Blood Pressure",
        "table" : f"""<table>
                <tr>
                    <th>Parameter</th>
                    <th>Mean ± SD</th>
                    <th>Range</th>
                    <th>Normality</th>
                </tr>
                <tr>
                    <td class = "param-name">{ap_hi.full_name} ({ap_hi.unit})</td>
                    <td>{mean_sd_range(df, ap_hi)[0]}</td>
                    <td>{mean_sd_range(df, ap_hi)[1]}</td>
                    <td>{pval_shapiro(df, ap_hi)}</td>
                </tr>
                <tr>
                    <td class = "param-name">{ap_lo.full_name} ({ap_lo.unit})</td>
                    <td>{mean_sd_range(df, ap_lo)[0]}</td>
                    <td>{mean_sd_range(df, ap_lo)[1]}</td>
                    <td>{pval_shapiro(df, ap_lo)}</td>
                </tr>
            </table>
            <br>
            <table>
            <tr>
                <th>Parameter</th>
                <th>Modalities</th>
                <th>Observations (<i>n</i>)</th>
                <th>Observations (%)</th>
            </tr>
            <tr>
                <td rowspan = "4" class = "param-name">{ap_aha.full_name}</td>
                <td>{ap_aha.mod_names[0]}</td>
                <td>{df[ap_aha.name].value_counts()[ap_aha.mod[0]]}</td>
                <td>{df[ap_aha.name].value_counts(normalize = True)[ap_aha.mod[0]]:.1%}</td>
            </tr>
            <tr>
                <td>{ap_aha.mod_names[1]}</td>
                <td>{df[ap_aha.name].value_counts()[ap_aha.mod[1]]}</td>
                <td>{df[ap_aha.name].value_counts(normalize = True)[ap_aha.mod[1]]:.1%}</td> 
            </tr>
            <tr>
                <td>{ap_aha.mod_names[2]}</td>
                <td>{df[ap_aha.name].value_counts()[ap_aha.mod[2]]}</td>
                <td>{df[ap_aha.name].value_counts(normalize = True)[ap_aha.mod[2]]:.1%}</td>
            </tr>
            <tr>
                <td>{ap_aha.mod_names[3]}</td>
                <td>{df[ap_aha.name].value_counts()[ap_aha.mod[3]]}</td>
                <td>{df[ap_aha.name].value_counts(normalize = True)[ap_aha.mod[3]]:.1%}</td>
            </tr>
        </table><br>""",
        "img" : "bp_fig.png",
        "analysis" : f"""<p>Both systolic and diastolic blood pressure displayed <b>substantial variability</b>, with wide range covered; and
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
            </p>""",
        "conclusion" : f"""<ul>
            <li>Only {df['ap_aha'].value_counts(normalize=True)['1']:.1%} of subjects had blood pressure classified as "Normal"</li>
            <li>Males displayed higher blood pressure values than females, regardless of cardiovascular status</li>
            <li>Blood pressure increased with age among controls, but remained almost constant in patients</li>
            <li>Prevalence of CV disease increased with AHA's categories, up to {cardio_aha[cardio_aha['ap_aha'] == "4"]['cardio_1'].values[0]:.1f}% among subjects
            in the "Hypertension Stage II" group</li>
        </ul>"""
    },
    "gluc-chol" : {
        "title" : "Glucose and Cholesterol",
        "table" : f"""<table>
            <tr>
                <th>Parameter</th>
                <th>Modalities</th>
                <th>Observations (<i>n</i>)</th>
                <th>Observations (%)</th>
            </tr>
            <tr>
                <td rowspan = "3" class = "param-name">{cholesterol.full_name}</td>
                <td>{cholesterol.mod_names[0]}</td>
                <td>{df[cholesterol.name].value_counts()[cholesterol.mod[0]]}</td>
                <td>{df[cholesterol.name].value_counts(normalize = True)[cholesterol.mod[0]]:.1%}</td>
            </tr>
            </tr>
            <tr>
                <td>{cholesterol.mod_names[1]}</td>
                <td>{df[cholesterol.name].value_counts()[cholesterol.mod[1]]}</td>
                <td>{df[cholesterol.name].value_counts(normalize = True)[cholesterol.mod[1]]:.1%}</td>
            </tr>   
            </tr>
            <tr>
                <td>{cholesterol.mod_names[2]}</td>
                <td>{df[cholesterol.name].value_counts()[cholesterol.mod[2]]}</td>
                <td>{df[cholesterol.name].value_counts(normalize = True)[cholesterol.mod[2]]:.1%}</td>
            </tr>
            </tr>
            <tr>
                <td rowspan = "3" class = "param-name">{gluc.full_name}</td>
                <td>{gluc.mod_names[0]}</td>
                <td>{df[gluc.name].value_counts()[gluc.mod[0]]}</td>
                <td>{df[gluc.name].value_counts(normalize = True)[gluc.mod[0]]:.1%}</td>
            </tr>
            </tr>
            <tr>
                <td>{gluc.mod_names[1]}</td>
                <td>{df[gluc.name].value_counts()[gluc.mod[1]]}</td>
                <td>{df[gluc.name].value_counts(normalize = True)[gluc.mod[1]]:.1%}</td>
            </tr>   
            </tr>
            <tr>
                <td>{gluc.mod_names[2]}</td>
                <td>{df[gluc.name].value_counts()[gluc.mod[2]]}</td>
                <td>{df[gluc.name].value_counts(normalize = True)[gluc.mod[2]]:.1%}</td>
            </tr>
        </table><br>""",
    "img" : "gluc_chol_fig.png",
    "analysis" : f"""<p>
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
                the feature <b>gluc</b> may bring limited information due to the influence of <b>cholesterol</b>.
            </p>""",
        "conclusion" : f"""<ul>
                <li>Majority of subjects had normal values for both glucose and cholesterol ({tab_gc.loc["1", "1"]:.1%})</li>
                <li>Prevalence of cardiovascular diseases increased with glucose and cholesterol</li>
                <li>Modalities "Above normal" and "Well above normal" could be merged for gluc</li>
                <li>Dropping gluc may be considered as it may bring a limited amount of information compared to 
                cholesterol</li>
                <li>Cut-off values for both gluc and cholesterol modalities are not known, there could
                be some bias, they therefore need to be interpreted with caution</li>
            </ul>"""
    },
    "lifestyle" : {
        "title" : "Lifestyle",
        "table" : f"""<table>
            <tr>
                <th>Parameter</th>
                <th>Modalities</th>
                <th>Observations (<i>n</i>)</th>
                <th>Observations (%)</th>
            </tr>
            <tr>
                <td rowspan = "2" class = "param-name">{smoke.full_name}</td>
                <td>{smoke.mod_names[0]}</td>
                <td>{df[smoke.name].value_counts()[smoke.mod[0]]}</td>
                <td>{df[smoke.name].value_counts(normalize = True)[smoke.mod[0]]:.1%}</td>
            </tr>
            <tr>
                <td>{smoke.mod_names[1]}</td>
                <td>{df[smoke.name].value_counts()[smoke.mod[1]]}</td>
                <td>{df[smoke.name].value_counts(normalize = True)[smoke.mod[1]]:.1%}</td>
            </tr>   
            <tr>
                <td rowspan = "2" class = "param-name">{alco.full_name}</td>
                <td>{alco.mod_names[0]}</td>
                <td>{df[alco.name].value_counts()[alco.mod[0]]}</td>
                <td>{df[alco.name].value_counts(normalize = True)[alco.mod[0]]:.1%}</td>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td>{alco.mod_names[1]}</td>
                <td>{df[alco.name].value_counts()[alco.mod[1]]}</td>
                <td>{df[alco.name].value_counts(normalize = True)[alco.mod[1]]:.1%}</td>
            </tr> 
            <tr>
                <td rowspan = "2" class = "param-name">{active.full_name}</td>
                <td>{active.mod_names[0]}</td>
                <td>{df[active.name].value_counts()[active.mod[0]]}</td>
                <td>{df[active.name].value_counts(normalize = True)[active.mod[0]]:.1%}</td>
            </tr>
            <tr>
                <td>{active.mod_names[1]}</td>
                <td>{df[active.name].value_counts()[active.mod[1]]}</td>
                <td>{df[active.name].value_counts(normalize = True)[active.mod[1]]:.1%}</td>
            </tr> 
        </table><br>""",
        "img" : "lifestyle_fig.png",
        "analysis": f"""            <p>
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
            </p>""",
        "conclusion" : f"""<ul>
                <li>A small number of subjects reported smoking and or drinking alcohol, and these variables did not 
                display any obvious correlation with cardio</li>
                <li>Smoking or exercising did not influence blood pressure levels</li>
                <li>A tendency of higher CV disease prevalence could exist among subjects that do not exercise</li>
                <li>Males were more likely to smoke and drink alcohol, while a comparable proportion of males and females
                reported physical activity</li>
                <li>smoke should not be included in the model due to plausible bias</li>
            </ul>"""
    }
}