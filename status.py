import streamlit as st
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
import markdown
from pathlib import Path
from PIL import Image
import xlrd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from statannot import add_stat_annotation

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot

from bioinfokit.analys import stat



st.set_page_config(page_title="Status", layout="wide", initial_sidebar_state="expanded",)
sns.set_style("white")


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

    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.replace("/", "_")

#df.columns = df.columns.str.strip()
#df.columns = df.columns.str.replace(" ", "_")
#df.columns = df.columns.str.replace("/", "_")

st.write(df)

if not isinstance(df, str):
    describe_data = st.sidebar.checkbox("Display data report")
    choose_analysis = st.sidebar.selectbox("Choose analysis to perform:", ("None", "Chi-square test", "Student t-test (Mann-Whitney)",
                                             "One, Two-way ANOVA", "Repeated measure ANOVA", "Correlation", "Linear Regression", "Logistic regression", 
                                             "Ridge, Lasso, Elastic net regression", "K-nearest neighbors", "Decision trees", 
                                             "Random forest", "Support vector machines", "Naive Bayes", "Neural network", 
                                             "Hierarchical Clusteting", "K-means Clustering", "Linear discriminant analysis", 
                                             "Principal component analysis", "XGBoost"))

if describe_data:
    pr = ProfileReport(df, explorative=True)
    st_profile_report(pr)

if choose_analysis == "Chi-square test":
    cat_vars = df.select_dtypes(include=np.object).columns.tolist()

    y_var1 = st.sidebar.selectbox("Choose first categorical variable:", (cat_vars))
    y_var2 = st.sidebar.selectbox("Choose second categorical variable:", (cat_vars))
    expand = st.sidebar.beta_expander("More options")
    yates_correction = expand.checkbox("Use Yates correction?")
    move_counts = expand.slider("Adjust labels for counts on bar plot", 0.0, 0.5, 0.15, 0.01)
    


    st.header("Chi-sqaure test of independence:")
    st.markdown("----")
    st.success("Expected and observed frequences:")

    expected, observed, stats = pg.chi2_independence(df, x=y_var1, y=y_var2, correction=True if yates_correction == True else False)
    st.subheader("Expected")
    st.write(expected)
    st.subheader("Observed")
    st.write(observed)
    st.subheader("Chi-square test results:")
    st.write(stats.loc[[0]])
    st.markdown("----")

    st.success("Frequency bars are generated:")
    st.markdown("## ")
    fig = plt.figure(figsize=(12, 6))
    ax = sns.countplot(x=y_var1, hue=y_var2, data=df, palette="Set2")
    for p in ax.patches:
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+move_counts, p.get_height()+2))
    st.pyplot(fig)

if choose_analysis == "One, Two-way ANOVA":
    numeric_vars = df.select_dtypes(include=np.number).columns.tolist()
    cat_vars = df.select_dtypes(include=np.object).columns.tolist()
    cat_vars2 = df.select_dtypes(include=np.object).columns.tolist()
    cat_vars2.insert(0, "None")



    x_var = st.sidebar.selectbox("Choose numetic variable:", (numeric_vars))
    y_var = st.sidebar.selectbox("Choose grouping variable:", (cat_vars))
    y_var2 = st.sidebar.selectbox("Choose second grouping variable:", (cat_vars2))

    expand = st.sidebar.beta_expander("More options")
    classic_vs_welch = expand.radio("Classic ANOVA or Welch ANOVA?", ("Classic ANOVA", "Welch ANOVA"))
    normality_selected = expand.radio("Select normality test", ["Shapiro-Wilk", "Omnibus test of normality"])
    error_selected = expand.radio("Choose error bar to plot", ["Standard error of the mean", "95% confidense intervals", "Standard deviation"])
    bars_width = expand.slider("Choose bar width", 0.0, 1.5, 0.5, 0.05)
    cap_selected = expand.checkbox("Plot caps on error bars")
    show_multiple = expand.checkbox("Show multiple comparisons on chart?")
    multiple_location = expand.radio("Multiple comparisons outside or inside the graph?", ("outside", "inside"))
    pvalue_text = expand.radio("Choose type of multiple comparison label", ("star", "simple", "full"))
    color_palette = expand.selectbox("Choose color palette", ("Set2", "Accent", "Blues", "BrBG", "Dark2", "GnBu", "Greys", "Oranges", "Paired",
                                        "Pastel1", "Purples", "Set1", "Set3", "Spectral", "Wistia", "autumn", "binary", "cividis", "cool", 
                                        "coolwarm", "icefire", "inferno", "magma", "ocean", "plasma", "rainbow", "summer", "twilight", "viridis", "winter"))



    

    st.header("Difference in means between groups results")
    st.success("Descriptive statistics are being calculated")
    function_dict = {x_var: ["mean", "std", "sem", "count"]}
    if y_var2 != "None":
        new = pd.DataFrame(df.groupby([y_var, y_var2]).aggregate(function_dict))
    else:
        new = pd.DataFrame(df.groupby(y_var).aggregate(function_dict))

    st.write(new)

    if normality_selected == "Shapiro-Wilk":
            message = "Shapiro-Wilk Normality test is being performed"
    else:
        message = "Omnibus test of normality is being performed"
   
    if y_var2 == "None":
        st.success(message)
    
    normality = pg.normality(df, dv=x_var, group=y_var, method="normaltest" if normality_selected == "Omnibus test of normality" else "shapiro")
    
    if y_var2 == "None":
        st.write(normality)

    if y_var2 == "None":
        st.success("Levene test for homoscedasticity of variances")

    homoscedasticity = pg.homoscedasticity(df, dv=x_var, group=y_var)
    if y_var2 == "None":
        st.write(homoscedasticity)

    if classic_vs_welch == "Classic ANOVA":
       
   
      
        if y_var2 == "None":
            anova = pg.anova(dv=x_var, between=y_var, data=df, detailed=True)
            st.success("Classic ANOVA results")
        else:
            
            anova = df.anova(dv=x_var, between=[y_var, y_var2])
            st.success("Two-way ANOVA results")
            

        st.write(anova.round(3))

        if y_var2 == "None":
            st.success("Tukey HSD multiple comparisons")
            tukey_mult = df.pairwise_tukey(dv=x_var, between=y_var).round(3)
            st.write(tukey_mult)

        if y_var2 != "None":
            st.success("Tukey HSD multiple comparisons")
            res = stat()
            st.subheader("Multiple comparisons of the 1st grouping variable")
            res.tukey_hsd(df=df, res_var=x_var, xfac_var=y_var, anova_model=f"{x_var}~C({y_var})+C({y_var2})+C({y_var}):C({y_var2})")
            st.write(res.tukey_summary)
            st.subheader("Multiple comparisons of the 2nd grouping variable")
            res.tukey_hsd(df=df, res_var=x_var, xfac_var=y_var2, anova_model=f"{x_var}~C({y_var})+C({y_var2})+C({y_var}):C({y_var2})")
            st.write(res.tukey_summary)
            st.subheader("Multiple comparisons of interactions")
            res.tukey_hsd(df=df, res_var=x_var, xfac_var=[y_var, y_var2], anova_model=f"{x_var}~C({y_var})+C({y_var2})+C({y_var}):C({y_var2})")
            st.write(res.tukey_summary)
            df_filtered = res.tukey_summary[res.tukey_summary['p-value'] < 0.05][['group1', 'group2']]
            tuples = [tuple(x) for x in df_filtered.to_numpy()]
            


            new_df = df.groupby([y_var,y_var2])[x_var].mean().reset_index()

            
            
            
            st.success("Interaction plot")
            fig = plt.figure(figsize=(12,6))
            sns.lineplot(data=new_df, x=y_var, y=x_var,
                 hue=y_var2, marker="d")

            st.pyplot(fig)

            

            
        if y_var2 == "None":
            df_filtered = tukey_mult[tukey_mult['p-tukey'] < 0.05][['A', 'B']]
            tuples = [tuple(x) for x in df_filtered.to_numpy()]
        


    else:
        test_message = "Welch ANOVA results:"
        st.success(test_message)

        welch = pg.anova(dv=x_var, between=y_var, data=df)
        st.write(welch.round(3))
        st.success("Games-Howell multiple comparisons")
        games_howell = pg.pairwise_gameshowell(dv=x_var, between=y_var, data=df).round(3)
        st.write(games_howell)
        df_filtered = games_howell[games_howell['pval'] < 0.05][['A', 'B']]
        tuples = [tuple(x) for x in df_filtered.to_numpy()]
        

    st.markdown("## ")
    
    st.success("Bar plots with errors are being generated")
    fig = plt.figure(figsize=(12,6))
    error = None
    if error_selected == "95% confidense intervals":
        error = 95
    elif error_selected == "Standard error of the mean":
        error = 68
    else:
        error =  "sd"

    if y_var2 == "None":
        ax = sns.barplot(x=y_var, y=x_var, data=df, ci=error , capsize=0.08 if cap_selected else 0, palette=color_palette, errwidth=0.7)
    else:
        

        ax = sns.barplot(x=y_var , y=x_var,
                 hue=y_var2, data=df, ci=error , 
                 capsize=0.08 if cap_selected else 0, palette=color_palette, errwidth=0.7)

    if y_var2 == "None":
        groups = len(df.groupby(y_var))
        widthbars = [bars_width] * groups
        for bar, newwidth in zip(ax.patches, widthbars):
            x = bar.get_x()
            width = bar.get_width()
            centre = x + width/2.
            bar.set_x(centre - newwidth/2.)
            bar.set_width(newwidth)

    if show_multiple and y_var2 == "None":
        ax, test_results = add_stat_annotation(ax, data=df, x=x_var, y=y_var, box_pairs=tuples,
                                   test='t-test_ind', text_format=pvalue_text, loc=multiple_location)
    if show_multiple and y_var2 != "None":
        ax, test_results = add_stat_annotation(ax, data=df, x=x_var, y=y_var, hue=y_var2, box_pairs=tuples,
                                   test='t-test_ind', text_format=pvalue_text, loc=multiple_location)


    st.pyplot(fig)

    
    fig = plt.figure(figsize=(12, 6))

    if y_var2 == "None":
        ax = sns.boxplot(x=y_var, y=x_var, data=df, width=.3, palette=color_palette)
        sns.swarmplot(x=y_var, y=x_var, data=df, size=3, color=".3", linewidth=0)
    else:
        ax = sns.boxplot(x=y_var, y=x_var, hue=y_var2, data=df, palette=color_palette)
    
    if show_multiple and y_var2=="None":
        ax, test_results = add_stat_annotation(ax, data=df, x=x_var, y=y_var, box_pairs=tuples,
                                   test='t-test_ind', text_format=pvalue_text, loc=multiple_location)
    if show_multiple and y_var2 != "None":
        ax, test_results = add_stat_annotation(ax, data=df, x=x_var, y=y_var, hue=y_var2, box_pairs=tuples,
                                   test='t-test_ind', text_format=pvalue_text, loc=multiple_location)
    

    st.pyplot(fig)



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

    st.header("Difference in means between groups results")
    st.success("Descriptive statistics are being calculated")
    function_dict = {x_var: ["mean", "std", "sem", "count"]}
    new = pd.DataFrame(df.groupby(y_var).aggregate(function_dict))
    st.write(new)

    if normality_selected == "Shapiro-Wilk":
        message = "Shapiro-Wilk Normality test is being perform"
    else:
        message = "Omnibus test of normality is being performed"
   
    st.success(message)
    
    normality = pg.normality(df, dv=x_var, group=y_var, method="normaltest" if normality_selected == "Omnibus test of normality" else "shapiro")
    st.write(normality)

    x1, x2 = df.groupby(y_var)[x_var].apply(list)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1 = pg.qqplot(x1, ax=ax1)
    ax2 = pg.qqplot(x2, ax=ax2)
    st.pyplot(fig)

    
    
    st.success("Levene test for homoscedasticity of variances")
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
    
    st.success("Bar plots with errors are being generated")
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

    st.header("Statistical methods being used")

    st.markdown("Statistical analysis was performed using Status software (https://status-please.herokuapp.com/)")
    st.markdown("Shapiro-Wilk test was used to estimate the normal distribution of the data. Levene test was used to check for equality of variances.")
    st.markdown("Student t-test was used to estimate the difference in means between groups in case of homoscedasticity. Welch test was used when variances were not equal.")
    st.markdown("Nonparametric Mann-Whitney test was used when data were not normally distributed.")
    st.markdown("Statistically significant results are considered with p-value < 0.05.")

    if show_anova_roadmap:
        image = Image.open("Images/anova.png")
        st.image(image)

   
