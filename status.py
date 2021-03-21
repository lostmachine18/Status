import streamlit as st
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
import markdown
from pathlib import Path

from PIL import Image

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
    choose_analysis = st.sidebar.selectbox("Choose analysis to perform:", ("None", "Chi-square test", "Student t-test (Mann-Whitney)", "One-way ANOVA", "Repeated measure ANOVA", "Linear Regression", "Logistic regression", "Ridge, Lasso, Elastic net regression", "K-nearest neighbors", "Decision trees", "Random forest", "Support vector machines", "Naive Bayes", "Neural network", "Hierarchical Clusteting", "K-means Clustering", "Linear discriminant analysis", "Principal component analysis", "XGBoost"))

if describe_data:
    pr = ProfileReport(df, explorative=True)
    st_profile_report(pr)

if choose_analysis == "Student t-test (Mann-Whitney)":

    numeric_vars = df.select_dtypes(include=np.number).columns.tolist()
    cat_vars = df.select_dtypes(include=np.object).columns.tolist()

    new_cat = []
    for item in cat_vars:
        if df.groupby(item).ngroups == 2:
            new_cat.append(item)
        
  
    x_var = st.sidebar.selectbox("Choose numetic variable:", (numeric_vars))
    y_var = st.sidebar.selectbox("Choose grouping variable:", (new_cat))

    expand = st.sidebar.beta_expander("More options")
    param_vs_nonparam = expand.radio("Parametric or nonparametric tests?", ("Parametric tests (Student, Welch)", "Nonparametric (Mann-Whitney)"))
    normality_selected = expand.radio("Select normality test", ["Shapiro-Wilk", "Omnibus test of normality"])
    error_selected = expand.radio("Choose error bar to plot:", ["Standard error of the mean", "95% confidense intervals", "Standard deviation"])
    cap_selected = expand.checkbox("Plot caps on error bars")
    show_boxplot = expand.checkbox("Show additional boxplots")
    show_anova_roadmap = expand.checkbox("Show roadmap for means analysis")

    st.header("Difference in means between groups results:")
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

    if param_vs_nonparam == "Parametric tests (Student, Welch)":
        if homoscedasticity.loc["levene", "pval"] < 0.05:
            test_message = "Welch test results:"
        else:
            test_message = "Student t-test results:"

        st.success(test_message)
        
        t = pg.ttest(x1, x2)
        st.write(t)

    else:
        test_message = "Mann-Whitney test results:"
        st.success(test_message)

        mw = pg.mwu(x1, x2)
        st.write(mw)
    
   
    md = markdown.Markdown()
    ipsum_path = Path('Md/student_help.md')

    data = ipsum_path.read_text(encoding='utf-8')
    html = md.convert(data)
    #help_markdown = util.read_markdown_file("help.md")
    st.markdown(html, unsafe_allow_html=True)


    
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

    ax = sns.barplot(x=y_var, y=x_var, data=df, ci=error , capsize=0.1 if cap_selected else 0, palette="Set2")
    widthbars = [0.4, 0.4]
    for bar, newwidth in zip(ax.patches, widthbars):
        x = bar.get_x()
        width = bar.get_width()
        centre = x + width/2.
        bar.set_x(centre - newwidth/2.)
        bar.set_width(newwidth)
    
    
    st.pyplot(fig)

    if show_boxplot:
        fig = plt.figure(figsize=(12, 6))
        sns.boxplot(x=x_var, y=y_var, data=df, width=.3, orient="h", palette="Set2")
        sns.stripplot(x=x_var, y=y_var, data=df, size=3, color=".3", linewidth=0)
        st.pyplot(fig)

    st.header("Statistical methods being used:")

    st.markdown("Statistical analysis was performed using Status software (https://status-please.herokuapp.com/)")
    st.markdown("Shapiro-Wilk test was used to estimate the normal distribution of the data. Levene test was used to check for equality of variances.")
    st.markdown("Student t-test was used to estimate the difference in means between groups in case of homoscedasticity. Welch test was used when variances were not equal.")
    st.markdown("Nonparametric Mann-Whitney test was used when data were not normally distributed.")
    st.markdown("Statistically significant results are considered with p-value < 0.05.")

    if show_anova_roadmap:
        image = Image.open("Images/anova.png")
        st.image(image)

   
