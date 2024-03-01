viz_data = {
    "1" : {
        "title": "Age",
        "table": """<table class = "table_1" style = "width: 75% !important">
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
            </table><br>""",
        "img": "age_fig.png",
        "analysis": """<p>
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
            </p>""",
        "conclusion": """<ul>
                <li>The mean age in the cohort was {mean_sd_range(df, age)[0]} yo.</li>
                <li>Age range was {mean_sd_range(df, age)[1]} yo: no children and no elderly people were included</li>
                <li>Patients in the cohort were older than controls, and the prevalence of CV disease linearly increased with age</li>
                <li>It could be worth adjusting the model on age, <i>ie</i> separating the data in two distinct dataframes with a cut-off value for age nearby 50 yo.</li>
            </ul>""",
    }
}