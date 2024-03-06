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
                <tr class = "head-tr">
                    <th>Parameter</th>
                    <th>Mean ± SD</th>
                    <th>Range</th>
                    <th>Normality</th>
                </tr>
                <tr>
                    <td><span class = "p-name">{age.full_name}</span> <span class = "p-unit">({age.unit})</span></td>
                    <td class = "value-td">{mean_sd_range(df, age)[0]}</td>
                    <td class = "value-td">{mean_sd_range(df, age)[1]}</td>
                    <td class = "value-td">{pval_shapiro(df, age)}</td>
                </tr>
            </table><br>""",
        "img": "age_fig.png",
        "analysis": f"""
        <p class="analysis-p">
            The dataset revealed a minimum age of {np.min(df_raw['age']) / 365.25:.1f} years, and the {len(df_raw[df_raw['age'] / 365.25 < 32])}
            individuals under 32 years were identified as outliers. There were no recorded cases of cardiovascular disease among the [32 - 38] years range.
            Given the significant age gap, these outliers were excluded from further analysis.
        </p>
        <p class="analysis-p">
            The adjusted age range for analysis spans from {np.min(df[age.name]):.0f} to {np.max(df[age.name]):.0f} years.
            The observed maximum age is notably lower than expected, considering cardiovascular diseases predominantly affect the elderly. 
            The reason for this absence of data from older individuals remains unclear but may stem from specific study protocol criteria.
        </p>
        <p class="analysis-p">
            As anticipated, the prevalence of cardiovascular (CV) diseases escalates with age within this cohort. Approximately 50% of patients aged [51 - 55] years exhibit CV diseases. 
            This prevalence dips below 50% among younger participants and increases for older age groups. The average age among patients diagnosed with CV diseases 
            is {mean_sd_range(df[df['cardio'] == "1"], age)[0]} years, in contrast to {mean_sd_range(df[df['cardio'] == "0"], age)[0]} years in the control group.
        </p>""",
        "conclusion": f"""<ul class="viz-cl">
            <li>The average age within the cohort was {mean_sd_range(df, age)[0]} years old.</li>
            <li>The age range was {mean_sd_range(df, age)[1]} years old, explicitly excluding children and the elderly from the study.</li>
            <li>Comparatively, patients were older than control subjects, with a clear linear increase in cardiovascular disease prevalence with advancing age.</li>
            <li>Consideration for age adjustment in the model might be beneficial, <i>e.g.</i>, dividing the dataset into two distinct groups using an age cut-off around 50 years old could provide deeper insights.</li>
        </ul>""",
    },
    "sex" : {
        "title" : "Sex",
        "table" : f"""<table>
            <tr class = "head-tr">
                <th>Parameter</th>
                <th>Modalities</th>
                <th>Observations (<span class = "obs">n</span>)</th>
                <th>Observations (<span class = "obs">%</span>)</th>
            </tr>
            <tr>
                <td rowspan = "2"><span class = "p-name">{sex.full_name}</span></td>
                <td class = "modality-td">{sex.mod[0]}</td>
                <td class = "value-td">{df[sex.name].value_counts()[sex.mod[0]]}</td>
                <td class = "value-td">{df[sex.name].value_counts(normalize = True)[sex.mod[0]]:.1%}</td>
            </tr>
            </tr>
            <tr>
                <td class = "modality-td">{sex.mod[1]}</td>
                <td class = "value-td">{df[sex.name].value_counts()[sex.mod[1]]}</td>
                <td class = "value-td">{df[sex.name].value_counts(normalize = True)[sex.mod[1]]:.1%}</td>
            </tr>
        </table><br>""",
        "img" : "sex_fig.png",
        "analysis" : f"""
        <p class="analysis-p">
            The cohort exhibited a significant sex imbalance, with approximately <sup>2</sup>/<sub>3</sub> being women and <sup>1</sup>/<sub>3</sub> men. 
            This disparity is notable within this age group and deviates from the expected sex ratio observed in geriatric populations.
        </p>
        <p class="analysis-p">
            Despite the imbalance, the distribution of cardiovascular disease (CV) between men and women appears proportionate, showing a prevalence of 
            {df[df['sex'] == 'female']['cardio'].value_counts(normalize=True)["1"]:.1%} in women and 
            {df[df['sex'] == 'male']['cardio'].value_counts(normalize=True)["1"]:.1%} in men. No significant link between sex and CV disease was identified according to chi² contingency tests ({chi2_cardio(df, sex)}). However, it's essential to consider that the similar proportions of Controls and Patients in each sex group might have been a study design criterion, potentially introducing bias into this observation.
        </p>
        <p class="analysis-p">
            Age distribution across genders was notably consistent, with females presenting a mean age of {np.mean(df[df['sex'] == 'female']['age'])/365.25:.1f} ± 
            {np.std(df[df['sex'] == 'female']['age'])/365.25:.1f} years, compared to males, who had a mean age of {np.mean(df[df['sex'] == 'male']['age'])/365.25:.1f} ± {np.std(df[df['sex'] == 'male']['age'])/365.25:.1f} years. Furthermore, the increase in CV disease prevalence with age was paralleled between the sexes.
        </p>""",
    "conclusion": f"""
        <ul class="viz-cl">
            <li>The influence of sex on cardiovascular disease prevalence seems negligible, with prevalence rates in both sex groups approximating 50%.</li>
            <li>Age-related increase in CV disease prevalence is consistent between males and females, indicating that age is a significant factor for CV disease risk irrespective of sex.</li>
            <li>Considering the disproportionate female-to-male ratio of nearly 2:1 in the study population, recalibrating the dataset for an equal sex distribution or creating gender-specific models could potentially refine predictive accuracy.</li>
        </ul>"""
    },
    "bmi" : {
        "title" : "Body-Mass Index, Height and Weight",
        "table" : f"""<table>
                <tr class = "head-tr">
                    <th>Parameter</th>
                    <th>Mean ± SD</th>
                    <th>Range</th>
                    <th>Normality</th>
                </tr>
                <tr>
                    <td><span class = "p-name">{bmi.full_name}</span> <span class = "p-unit">({bmi.unit})</span></td>
                    <td class = "value-td">{mean_sd_range(df, bmi)[0]}</td>
                    <td class = "value-td">{mean_sd_range(df, bmi)[1]}</td>
                    <td class = "value-td">{pval_shapiro(df, bmi)}</td>
                </tr>
                <tr>
                    <td><span class = "p-name">{weight.full_name}</span> <span class = "p-unit">({weight.unit})</span></td>
                    <td class = "value-td">{mean_sd_range(df, weight)[0]}</td>
                    <td class = "value-td">{mean_sd_range(df, weight)[1]}</td>
                    <td class = "value-td">{pval_shapiro(df, weight)}</td>
                </tr>
                <tr>
                    <td><span class = "p-name">{height.full_name}</span> <span class = "p-unit">({height.unit})</span></td>
                    <td class = "value-td">{mean_sd_range(df, height)[0]}</td>
                    <td class = "value-td">{mean_sd_range(df, height)[1]}</td>
                    <td class = "value-td">{pval_shapiro(df, height)}</td>
                </tr>
            </table><br>""",
        "img": "bmi_height_weight_fig.png",
        "analysis": f"""
        <p class="analysis-p">
            The dataset spans a broad spectrum of physical measurements, with heights 
            ranging from {mean_sd_range(df, height)[1]} cm and weights 
            from {mean_sd_range(df, weight)[1]} kg. This diversity yields a
            Body Mass Index (BMI) range of {mean_sd_range(df, bmi)[1]} kg/m², 
            centering around a median BMI of {np.median(df['bmi']):.1f} kg/m². 
            Notably, while men averaged taller than women, BMI distribution remained 
            consistent across sexes.
        </p>
        <p class="analysis-p">
            Analysis indicates a clear correlation: the prevalence of cardiovascular 
            diseases escalates with increasing BMI. Extreme weight and height measurements
            demonstrated significant variability, particularly pronounced at higher 
            weight ranges.
        </p>
        <p class="analysis-p">
            Cardiovascular disease affected approximately 50% of individuals whose 
            height and weight fell within their respective 95% confidence 
            intervals ([{h_lo} - {h_hi}] cm for height 
            and [{w_lo} - {w_hi}] kg for weight). Among those with extreme 
            measurements, weight appeared to exert a more pronounced impact on 
            cardiovascular disease risk than height.
        </p>""",
    "conclusion": f"""
        <ul class="viz-cl">
            <li>The prevalence of cardiovascular diseases increases with BMI.</li>
            <li>The BMI distribution in this cohort skews right, peaking at {np.max(df['bmi']):.1f} kg/m². However, the influence of outliers appears minimal.</li>
            <li>In cases involving outlier values for height and weight, the latter demonstrates a more pronounced impact on cardiovascular risk.</li>
            <li>Evaluating the effect of extreme height and weight measurements on the model's performance could provide valuable insights.</li>
        </ul>"""
    },
    "bp": {
        "title" : "Blood Pressure",
        "table" : f"""<table>
                <tr class = "head-tr">
                    <th>Parameter</th>
                    <th>Mean ± SD</th>
                    <th>Range</th>
                    <th>Normality</th>
                </tr>
                <tr>
                    <td><span class = "p-name">{ap_hi.full_name}</span> <span class = "p-unit">({ap_hi.unit})</span></td>
                    <td class = "value-td">{mean_sd_range(df, ap_hi)[0]}</td>
                    <td class = "value-td">{mean_sd_range(df, ap_hi)[1]}</td>
                    <td class = "value-td">{pval_shapiro(df, ap_hi)}</td>
                </tr>
                <tr>
                    <td><span class = "p-name">{ap_lo.full_name}</span> <span class = "p-unit">({ap_lo.unit})</span></td>
                    <td class = "value-td">{mean_sd_range(df, ap_lo)[0]}</td>
                    <td class = "value-td">{mean_sd_range(df, ap_lo)[1]}</td>
                    <td class = "value-td">{pval_shapiro(df, ap_lo)}</td>
                </tr>
            </table>
            <br>
            <table>
            <tr class = "head-tr">
                <th>Parameter</th>
                <th>Modalities</th>
                <th>Observations (<span class = "obs">n</span>)</th>
                <th>Observations (<span class = "obs">%</span>)</th>
            </tr>
            <tr>
                <td rowspan = "4"><span class = "p-name">{ap_aha.full_name}</span></td>
                <td class = "modality-td">{ap_aha.mod_names[0]}</td>
                <td class = "value-td">{df[ap_aha.name].value_counts()[ap_aha.mod[0]]}</td>
                <td class = "value-td">{df[ap_aha.name].value_counts(normalize = True)[ap_aha.mod[0]]:.1%}</td>
            </tr>
            <tr>
                <td class = "modality-td">{ap_aha.mod_names[1]}</td>
                <td class = "value-td">{df[ap_aha.name].value_counts()[ap_aha.mod[1]]}</td>
                <td class = "value-td">{df[ap_aha.name].value_counts(normalize = True)[ap_aha.mod[1]]:.1%}</td> 
            </tr>
            <tr>
                <td class = "modality-td">{ap_aha.mod_names[2]}</td>
                <td class = "value-td">{df[ap_aha.name].value_counts()[ap_aha.mod[2]]}</td>
                <td class = "value-td">{df[ap_aha.name].value_counts(normalize = True)[ap_aha.mod[2]]:.1%}</td>
            </tr>
            <tr>
                <td class = "modality-td">{ap_aha.mod_names[3]}</td>
                <td class = "value-td">{df[ap_aha.name].value_counts()[ap_aha.mod[3]]}</td>
                <td class = "value-td">{df[ap_aha.name].value_counts(normalize = True)[ap_aha.mod[3]]:.1%}</td>
            </tr>
        </table><br>""",
        "img" : "bp_fig.png",
        "analysis" : f"""
            <p class="analysis-p">
                Both systolic and diastolic blood pressures exhibited substantial v
                ariability and were found to be closely correlated 
                ({pval_txt(spearmanr(df['ap_hi'], df['ap_lo'])[1])}). Notably, higher blood
                pressure values were observed in patients diagnosed with cardiovascular 
                disease-evident in systolic ({pval_txt(mwu_cardio(df, ap_hi))}),
                diastolic ({pval_txt(mwu_cardio(df, ap_lo))}), and mean blood pressures 
                ({pval_txt(mwu_cardio(df, ap_m))}). This aligns with the established 
                understanding that hypertension is a major risk factor for 
                cardiovascular conditions.
            </p>
            <p class="analysis-p">
                The disease prevalence escalated linearly across the American Heart 
                Association's categories from "Normal" through "Elevated" to 
                "Hypertension Stage 1", marking an 11% increase between
                these stages. Remarkably, prevalence almost doubled transitioning 
                from "Hypertension Stage I" ({cardio_aha[cardio_aha['ap_aha'] == "3"]['cardio_1'].values[0]:.1f}%)
                to "Hypertension Stage II" ({cardio_aha[cardio_aha['ap_aha'] == "4"]['cardio_1'].values[0]:.1f}%).
            </p>
            <p class="analysis-p">
                Overall, male subjects demonstrated higher mean blood pressure levels than female, independent
                of age or cardiovascular disease status. This disparity in blood pressure appeared to
                lessen with age among control subjects but persisted among patients.
                Moreover, while age did not significantly influence mean blood pressure in 
                patients, it was associated with an increase in controls.
            </p>""",
        "conclusion" : f"""
            <ul class="viz-cl">
            <li>Only {df['ap_aha'].value_counts(normalize=True)['1']:.1%} of participants were categorized within the 'Normal' blood pressure range, highlighting the prevalence of elevated levels within the cohort.</li>
            <li>Irrespective of cardiovascular disease status, male participants consistently showed higher blood pressure readings compared to females.</li>
            <li>While blood pressure readings tend to rise with age in control subjects, they remained relatively stable among patients with cardiovascular conditions.</li>
            <li>The incidence of cardiovascular diseases increases in alignment with the American Heart Association’s (AHA) categorizations, peaking at {cardio_aha[cardio_aha['ap_aha'] == "4"]['cardio_1'].values[0]:.1f}% within individuals classified under 'Hypertension Stage II'.</li>
        </ul>"""
    },
    "gluc-chol" : {
        "title" : "Glucose and Cholesterol",
        "table" : f"""<table>
            <tr class = "head-tr">
                <th>Parameter</th>
                <th>Modalities</th>
                <th>Observations (<span class = "obs">n</span>)</th>
                <th>Observations (<span class = "obs">%</span>)</th>
            </tr>
            <tr>
                <td rowspan = "3"><span class = "p-name">{cholesterol.full_name}</span></td>
                <td class = "modality-td">{cholesterol.mod_names[0]}</td>
                <td class = "value-td">{df[cholesterol.name].value_counts()[cholesterol.mod[0]]}</td>
                <td class = "value-td">{df[cholesterol.name].value_counts(normalize = True)[cholesterol.mod[0]]:.1%}</td>
            </tr>
            </tr>
            <tr>
                <td class = "modality-td">{cholesterol.mod_names[1]}</td>
                <td class = "value-td">{df[cholesterol.name].value_counts()[cholesterol.mod[1]]}</td>
                <td class = "value-td">{df[cholesterol.name].value_counts(normalize = True)[cholesterol.mod[1]]:.1%}</td>
            </tr>   
            </tr>
            <tr>
                <td class = "modality-td">{cholesterol.mod_names[2]}</td>
                <td class = "value-td">{df[cholesterol.name].value_counts()[cholesterol.mod[2]]}</td>
                <td class = "value-td">{df[cholesterol.name].value_counts(normalize = True)[cholesterol.mod[2]]:.1%}</td>
            </tr>
            </tr>
            <tr>
                <td rowspan = "3"<span class = "p-name">{gluc.full_name}</span></td>
                <td class = "modality-td">{gluc.mod_names[0]}</td>
                <td class = "value-td">{df[gluc.name].value_counts()[gluc.mod[0]]}</td>
                <td class = "value-td">{df[gluc.name].value_counts(normalize = True)[gluc.mod[0]]:.1%}</td>
            </tr>
            </tr>
            <tr>
                <td class = "modality-td">{gluc.mod_names[1]}</td>
                <td class = "value-td">{df[gluc.name].value_counts()[gluc.mod[1]]}</td>
                <td class = "value-td">{df[gluc.name].value_counts(normalize = True)[gluc.mod[1]]:.1%}</td>
            </tr>   
            </tr>
            <tr>
                <td class = "modality-td">{gluc.mod_names[2]}</td>
                <td class = "value-td">{df[gluc.name].value_counts()[gluc.mod[2]]}</td>
                <td class = "value-td">{df[gluc.name].value_counts(normalize = True)[gluc.mod[2]]:.1%}</td>
            </tr>
        </table><br>""",
    "img" : "gluc_chol_fig.png",
    "analysis" : f"""
        <p class="analysis-p">
            The vast majority of the cohort maintained normal levels for glucose 
            ({df["gluc"].value_counts(normalize=True)["1"]:.1%}) and cholesterol 
            ({df["cholesterol"].value_counts(normalize=True)["1"]:.1%}), with similar 
            proportions across both markers for those categorized as
            "Above normal" and "Well above normal."
        </p>
        <p class="analysis-p">
            Notably, the prevalence of cardiovascular disease escalates with elevated 
            glucose and cholesterol levels. Among individuals with normal 
            glucose levels, the prevalence was
            {df[df["gluc"] == "1"]['cardio'].value_counts(normalize=True)["1"]:.1%}, 
            and {df[df["cholesterol"] == "1"]['cardio'].value_counts(normalize=True)["1"]:.1%}
            for those with normal cholesterol levels. The disease prevalence increases linearly
            with cholesterol, reaching up to
            {df[df["cholesterol"] == "3"]['cardio'].value_counts(normalize=True)["1"]:.1%}
            in individuals with "Well above normal" cholesterol levels. Conversely, 
            glucose's impact is less differentiated, maintaining a near-constantprevalence
            between the "Above normal" and "Well above normal" categories 
            (approximately {np.mean([df[df["gluc"] == "2"]['cardio'].value_counts(normalize=True)["1"], df[df["gluc"] == "3"]['cardio'].value_counts(normalize=True)["1"]]):.0%}).
        </p>
        <p class="analysis-p">
            Concurrent analysis of glucose and cholesterol levels shows that a 
            significant fraction had normal readings for both ({tab_gc.loc["1", "1"]:.1%}).
            Intriguingly, cardiovascular disease prevalence does not markedly 
            vary with glucose levels when cholesterol remains constant, 
            suggesting that while both factors are associated with cardiovascular 
            health ({pval_chi2_gluc} for glucose and {pval_chi2_chol} for cholesterol),
            glucose may offer limited additional predictive value over cholesterol.
        </p>""",
        "conclusion" : f"""
            <ul class="viz-cl">
                <li>The majority of participants displayed normal levels for both glucose and cholesterol, with a significant proportion ({tab_gc.loc["1", "1"]:.1%}) falling within this healthy range.</li>
                <li>The likelihood of cardiovascular diseases escalates with higher levels of glucose and cholesterol, underscoring the importance of monitoring these biomarkers.</li>
                <li>Considering the minimal differentiation in cardiovascular disease prevalence between "Above normal" and "Well above normal" glucose levels, merging these categories for glucose analysis could streamline the dataset.</li>
                <li>The potential redundancy of glucose as a predictive factor warrants discussion, with the suggestion to potentially exclude it from analyses in favor of focusing on cholesterol, which appears to offer more substantial insights.</li>
                <li>Given the absence of clear cut-off values for defining glucose and cholesterol levels, caution is advised when interpreting these parameters due to the potential for bias in their categorization.</li>
            </ul>"""
    },
    "lifestyle" : {
        "title" : "Lifestyle",
        "table" : f"""<table>
            <tr class = "head-tr">
                <th>Parameter</th>
                <th>Modalities</th>
                <th>Observations (<span class = "obs">n</span>)</th>
                <th>Observations (<span class = "obs">%</span>)</th>
            </tr>
            <tr>
                <td rowspan = "2"><span class = "p-name">{smoke.full_name}</span></td>
                <td class = "modality-td">{smoke.mod_names[0]}</td>
                <td class = "value-td">{df[smoke.name].value_counts()[smoke.mod[0]]}</td>
                <td class = "value-td">{df[smoke.name].value_counts(normalize = True)[smoke.mod[0]]:.1%}</td>
            </tr>
            <tr>
                <td class = "modality-td">{smoke.mod_names[1]}</td>
                <td class = "value-td">{df[smoke.name].value_counts()[smoke.mod[1]]}</td>
                <td class = "value-td">{df[smoke.name].value_counts(normalize = True)[smoke.mod[1]]:.1%}</td>
            </tr>   
            <tr>
                <td rowspan = "2"><span class = "p-name">{alco.full_name}</span></td>
                <td class = "modality-td">{alco.mod_names[0]}</td>
                <td class = "value-td">{df[alco.name].value_counts()[alco.mod[0]]}</td>
                <td class = "value-td">{df[alco.name].value_counts(normalize = True)[alco.mod[0]]:.1%}</td>
            </tr>
            <tr style = 'border-top: 1px solid black'>
                <td class = "modality-td">{alco.mod_names[1]}</td>
                <td class = "value-td">{df[alco.name].value_counts()[alco.mod[1]]}</td>
                <td class = "value-td">{df[alco.name].value_counts(normalize = True)[alco.mod[1]]:.1%}</td>
            </tr> 
            <tr>
                <td rowspan = "2"><span class = "p-name">{active.full_name}</span></td>
                <td class = "modality-td">{active.mod_names[0]}</td>
                <td class = "value-td">{df[active.name].value_counts()[active.mod[0]]}</td>
                <td class = "value-td">{df[active.name].value_counts(normalize = True)[active.mod[0]]:.1%}</td>
            </tr>
            <tr>
                <td class = "modality-td">{active.mod_names[1]}</td>
                <td class = "value-td">{df[active.name].value_counts()[active.mod[1]]}</td>
                <td class = "value-td">{df[active.name].value_counts(normalize = True)[active.mod[1]]:.1%}</td>
            </tr> 
        </table><br>""",
        "img" : "lifestyle_fig.png",
        "analysis": f"""
            <p class="analysis-p">
                A small percentage of participants engaged in smoking 
                ({df['smoke'].value_counts(normalize=True)["1"]:.1%}) or alcohol consumption
                ({df['alco'].value_counts(normalize=True)["1"]:.1%}), while a significant 
                majority ({df['active'].value_counts(normalize=True)["1"]:.1%}) reported 
                participating in some form of physical activity. Consequently, a
                substantial portion of the cohort 
                ({df['healthy_ls'].value_counts(normalize=True)["1"]:.1%}) exhibited an 
                overall healthy lifestyle, characterized by abstaining from smoking, 
                moderate to no alcohol intake, and engaging in physical activity. 
                In contrast, a very small fraction 
                ({df['lifestyle'].value_counts(normalize=True)["7"]:.1%}) reported smoking, 
                drinking alcohol, and a lack of physical activity. It is important to note, 
                however, that the specific criteria for alcohol consumption and physical 
                activity levels are undefined, which could introduce bias into these 
                observations.
            </p>
            <p class="analysis-p">
                Intriguingly, the analysis revealed no significant impact of smoking or 
                physical activity on blood pressure levels, with median systolic, 
                diastolic, and mean blood pressure values showing no difference, 
                and their distributions remaining similar across groups.        
            </p>
            <p class="analysis-p">
                Sexe-related differences emerged in lifestyle habits, with male 
                participants more prone to smoking and alcohol consumption than  
                female: {df_sex_ls[(df_sex_ls['sex'] == 'male') & (df_sex_ls['parameter'] == "smoke_1")]['percentage'].values[0]:.1f}% 
                of males smoked, and {df_sex_ls[(df_sex_ls['sex'] == 'male') & (df_sex_ls['parameter'] == "alco_1")]['percentage'].values[0]:.1f}% 
                consumed alcohol, compared to {df_sex_ls[(df_sex_ls['sex'] == 'female') & (df_sex_ls['parameter'] == "smoke_1")]['percentage'].values[0]:.1f}% 
                and {df_sex_ls[(df_sex_ls['sex'] == 'female') & (df_sex_ls['parameter'] == "alco_1")]['percentage'].values[0]:.1f}% respectively for females.
            </p>""",
        "conclusion" : f"""
            <ul class="viz-cl">
                <li>A minority of the study's participants reported engaging in smoking or alcohol consumption, neither of which showed a clear correlation with cardiovascular disease incidence.</li>
                <li>Neither smoking habits nor physical activity levels appeared to significantly impact blood pressure readings.</li>
                <li>There may be a potential trend towards a higher prevalence of CV diseases among individuals who do not participate in regular physical activity.</li>
                <li>Male participants were more inclined towards smoking and alcohol consumption compared to females, yet both genders reported similar rates of physical activity engagement.</li>
                <li>Due to potential biases, incorporating the smoking variable into the predictive model may not be advisable.</li>
            </ul>"""
    }
}