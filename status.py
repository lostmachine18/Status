import streamlit as st
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit_theme as stt

st.set_page_config(page_title="Status", layout="wide", initial_sidebar_state="expanded",)
sns.set_style("white")
# sns.set(rc={'figure.figsize':(3, 2)})


import xlrd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

global df
describe_data = None
choose_analysis = None

st.title("Status")
st.markdown("## _Your personal status of data analysis_")
st.markdown("____")


uploaded_file = st.sidebar.file_uploader(
                        label="Choose your data",
                        type=['csv', 'xlsx'])

global df

df = "To start your work, please choose the data!"

if uploaded_file is not None:
    
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        print(e)
        df = pd.read_excel(uploaded_file)

st.write(df)

if not isinstance(df, str):
    describe_data = st.sidebar.checkbox("Display data report")
    choose_analysis = st.sidebar.selectbox("Choose analysis to perform:", ("None", "Chi-square test", "Student t-test", "One-way ANOVA", "Repeated measure ANOVA", "Linear Regression", "Logistic regression", "K-nearest neighbors", "Decision trees", "Random forest", "Support vector machines", "Neural network"))

if describe_data:
    pr = ProfileReport(df, explorative=True)
    st_profile_report(pr)

if choose_analysis == "Student t-test":

    numeric_vars = df.select_dtypes(include=np.number).columns.tolist()
    cat_vars = df.select_dtypes(include=np.object).columns.tolist()

    new_cat = []
    for item in cat_vars:
        if df.groupby(item).ngroups == 2:
            new_cat.append(item)
        
  
    x_var = st.sidebar.selectbox("Choose numetic variable:", (numeric_vars))
    y_var = st.sidebar.selectbox("Choose grouping variable:", (new_cat))

    expand = st.sidebar.beta_expander("More options")
    help_selected = expand.checkbox("Show additional help on t-test?")
    normality_selected = expand.radio("Select normality test", ["Shapiro-Wilk", "Omnibus test of normality"])
    error_selected = expand.radio("Choose error bar to plot:", ["Standard error of the mean", "95% confidense intervals", "Standard deviation"])
    cap_selected = expand.checkbox("Plot caps on error bars")

    st.header("Student t-test results:")
    st.success("Descriptive statistics are being calculated")
    function_dict = {x_var: ["mean", "std", "sem", "count"]}
    new = pd.DataFrame(df.groupby(y_var).aggregate(function_dict))
    st.write(new)

    if normality_selected == "Shapiro-Wilk":
        message = "Shapiro-Wilk Normality test is being perform:"
    else:
        message = "Omnibus test of normality is being performed:"
   
    st.success(message)
    
    normality = pg.normality(df, dv=x_var, group=y_var, method="normaltest" if normality_selected == "Omnibus test of normality" else "shapiro")
    st.write(normality)

    x1, x2 = df.groupby(y_var)[x_var].apply(list)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1 = pg.qqplot(x1, ax=ax1)
    ax2 = pg.qqplot(x2, ax=ax2)
    st.pyplot(fig)

    
    
    st.success("Levene test for homoscedasticity of variances:")
    homoscedasticity = pg.homoscedasticity(df, dv=x_var, group=y_var)
    st.write(homoscedasticity)


    if homoscedasticity.loc["levene", "pval"] < 0.05:
        test_message = "Welch test results:"
    else:
         test_message = "Student t-test results:"

    st.success(test_message)
    
    
    t = pg.ttest(x1, x2)
    st.write(t)
    if help_selected:
        st.markdown("_ Please, note, that Welch test is performed when heteroscedasticity is observed. Student test is performed when variances are equal. _")
        st.markdown(" _**T** is a Student statistic._")
        st.markdown("_**dof** means degrees of freedom._")
        st.markdown("_**tail** is usually set to two-sided which which that your hypothesis is tested in both directions._")
        st.markdown("_**pval** is statistical significance. It's largely dependent on sample size._")
        st.markdown("_**CI95%** are 95% confidence intervals for the test statistic._")
        st.markdown("_**cohen-d** is an effect size. 0.2 means small effect, 0.5 - medium, 0.8 - large._")
        st.markdown("_**power** is a statistical power of your test. More tha 80% is an acceptable rate._")

    st.markdown("## ")
    
    st.success("Bar plots with errors are being generated:")
    fig = plt.figure(figsize=(12,6))
    error = None
    if error_selected == "95% confidense intervals":
        error = 95
    elif error_selected == "Standard error of the mean":
        error = 68
    else:
        error =  "sd"

    sns.barplot(x=y_var, y=x_var, data=df, ci=error , capsize=0.1 if cap_selected else 0)
    
    st.pyplot(fig)

    st.header("Statistical methods being used:")

    st.markdown("Statistical analysis was performed using Status software (https://status-please.herokuapp.com/)")
    st.markdown("Shapiro-Wilk test was used to estimate the normal distribution of the data. Levene test was used to check for equality of variances.")
    st.markdown("Student t-test was used to estimate the difference in means between groups in case of homoscedasticity. Welch test was used when variances were not equal.")
    st.markdown("Statistically significant results are considered with p-value < 0.05.")

