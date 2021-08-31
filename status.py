import streamlit as st
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
import markdown
from pathlib import Path
from PIL import Image
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from statannot import add_stat_annotation
from math import sqrt


from bioinfokit.analys import stat

st.set_page_config(page_title="Status", layout="wide", initial_sidebar_state="expanded", )
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

df = "To start your work, please choose the data!"

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        print(e)
        df = pd.read_excel(uploaded_file, engine='openpyxl')

    df.columns.str.strip()
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.replace("/", "_")

st.write(df)

if not isinstance(df, str):
    describe_data = st.sidebar.checkbox("Display data report")
    choose_analysis = st.sidebar.selectbox("Choose analysis to perform:",
                                           ("None", "Count test", "Chi-square test", "Student t-test (Mann-Whitney)",
                                            "One, Two-way ANOVA", "Non-parametric ANOVA",
                                            "Repeated measures mixed ANOVA", "Correlation", "Linear Regression",
                                            "Logistic regression",
                                            "Ridge, Lasso, Elastic net regression", "K-nearest neighbors",
                                            "Decision trees",
                                            "Random forest", "Support vector machines", "Naive Bayes", "Neural network",
                                            "Hierarchical Clustering", "K-means Clustering",
                                            "Linear discriminant analysis",
                                            "Principal component analysis", "XGBoost"))

if describe_data:
    pr = ProfileReport(df, explorative=True)
    st_profile_report(pr)


if choose_analysis == "Count test":
    x1 = st.sidebar.number_input("Enter counts for the 1st variable", 0, value=5, step=1)
    x2 = st.sidebar.number_input("Enter counts for the 2nd variable", 0, value=10, step=1)

    st.success("Difference between treatments results")

    z = (int(x1) - int(x2)) / sqrt((int(x1) + int(x2)))
    st.write(f"z-score = {z}")

    p_value = "< 0.05" if abs(z) > 1.96 else "> 0.05"
    st.write(f"Statistical significance p {p_value}")

if choose_analysis == "Chi-square test":
    cat_vars = df.select_dtypes(include=np.object).columns.tolist()

    y_var1 = st.sidebar.selectbox("Choose first categorical variable:", cat_vars)
    y_var2 = st.sidebar.selectbox("Choose second categorical variable:", cat_vars)
    expand = st.sidebar.beta_expander("More options")
    yates_correction = expand.checkbox("Use Yates correction?")
    # move_counts = expand.slider("Adjust labels for counts on bar plot", 0.0, 0.5, 0.15, 0.01)

    st.header("Chi-square test of independence:")
    st.markdown("----")
    st.success("Expected and observed frequencies:")

    expected, observed, stats = pg.chi2_independence(df, x=y_var1, y=y_var2,
                                                     correction=True if yates_correction else False)
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
    total = float(len(df))
    ax = sns.countplot(x=y_var1, hue=y_var2, data=df, palette="Set2")
    numX=len([x for x in df[y_var1].unique() if x==x])

    # 2. The bars are created in hue order, organize them
    bars = ax.patches
    ## 2a. For each X variable
    for ind in range(numX):
        ## 2b. Get every hue bar
        ##     ex. 8 X categories, 4 hues =>
        ##    [0, 8, 16, 24] are hue bars for 1st X category
        hueBars=bars[ind:][::numX]
        ## 2c. Get the total height (for percentages)
        total = sum([x.get_height() for x in hueBars])

        # 3. Print the percentage on the bars
        for bar in hueBars:
            ax.text(bar.get_x() + bar.get_width()/2.,
                    bar.get_height(),
                    f'{bar.get_height()}({bar.get_height()/total:.0%})',
                    ha="center",va="bottom")
    #for p in ax.patches:
    #    ax.annotate('{:.0f}%'.format(p.get_height()), (p.get_x() + move_counts, p.get_height() + 2))
    st.pyplot(fig)

if choose_analysis == "Student t-test (Mann-Whitney)":

    numeric_vars = df.select_dtypes(include=np.number).columns.tolist()
    cat_vars = df.select_dtypes(include=np.object).columns.tolist()

    new_cat = []
    for item in cat_vars:
        if df.groupby(item).ngroups == 2:
            new_cat.append(item)

    x_var = st.sidebar.selectbox("Choose numeric variable:", numeric_vars)
    y_var = st.sidebar.selectbox("Choose grouping variable:", new_cat)

    expand = st.sidebar.beta_expander("More options")
    param_vs_nonparam = expand.radio("Parametric or nonparametric tests?",
                                     ("Parametric tests (Student, Welch)", "Nonparametric (Mann-Whitney)"))
    normality_selected = expand.radio("Select normality test", ["Shapiro-Wilk", "Omnibus test of normality"])
    error_selected = expand.radio("Choose error bar to plot:",
                                  ["Standard error of the mean", "95% confidence intervals", "Standard deviation"])
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

    normality = pg.normality(df, dv=x_var, group=y_var,
                             method="normaltest" if normality_selected == "Omnibus test of normality" else "shapiro")
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
    # help_markdown = util.read_markdown_file("help.md")
    st.markdown(html, unsafe_allow_html=True)

    st.markdown("## ")

    st.success("Bar plots with errors are being generated")
    fig = plt.figure(figsize=(12, 6))
    error = None
    if error_selected == "95% confidense intervals":
        error = 95
    elif error_selected == "Standard error of the mean":
        error = 68
    else:
        error = "sd"

    ax = sns.barplot(x=y_var, y=x_var, data=df, ci=error, capsize=0.1 if cap_selected else 0, palette="Set2")
    widthbars = [0.4, 0.4]
    for bar, newwidth in zip(ax.patches, widthbars):
        x = bar.get_x()
        width = bar.get_width()
        centre = x + width / 2.
        bar.set_x(centre - newwidth / 2.)
        bar.set_width(newwidth)

    st.pyplot(fig)

    if show_boxplot:
        fig = plt.figure(figsize=(12, 6))
        sns.boxplot(x=x_var, y=y_var, data=df, width=.3, orient="h", palette="Set2")
        sns.stripplot(x=x_var, y=y_var, data=df, size=3, color=".3", linewidth=0)
        st.pyplot(fig)

    st.header("Statistical methods being used")

    st.markdown("Statistical analysis was performed using Status software (https://status-please.herokuapp.com/)")
    st.markdown(
        "Shapiro-Wilk test was used to estimate the normal distribution of the data. Levene test was used to check "
        "for equality of variances.")
    st.markdown(
        "Student t-test was used to estimate the difference in means between groups in case of homoscedasticity. "
        "Welch test was used when variances were not equal.")
    st.markdown("Nonparametric Mann-Whitney test was used when data were not normally distributed.")
    st.markdown("Statistically significant results are considered with p-value < 0.05.")

    if show_anova_roadmap:
        image = Image.open("Images/anova.png")
        st.image(image)

if choose_analysis == "One, Two-way ANOVA":
    numeric_vars = df.select_dtypes(include=np.number).columns.tolist()
    cat_vars = df.select_dtypes(include=np.object).columns.tolist()
    cat_vars2 = df.select_dtypes(include=np.object).columns.tolist()
    cat_vars2.insert(0, "None")

    x_var = st.sidebar.selectbox("Choose numeric variable:", numeric_vars)
    y_var = st.sidebar.selectbox("Choose grouping variable:", cat_vars)
    y_var2 = st.sidebar.selectbox("Choose second grouping variable:", (cat_vars2))

    expand = st.sidebar.beta_expander("More options")
    classic_vs_welch = expand.radio("Classic ANOVA or Welch ANOVA?", ("Classic ANOVA", "Welch ANOVA"))
    normality_selected = expand.radio("Select normality test", ["Shapiro-Wilk", "Omnibus test of normality"])
    error_selected = expand.radio("Choose error bar to plot",
                                  ["Standard error of the mean", "95% confidence intervals", "Standard deviation"])
    bars_width = expand.slider("Choose bar width", 0.0, 1.5, 0.5, 0.05)
    cap_selected = expand.checkbox("Plot caps on error bars")
    show_multiple = expand.checkbox("Show multiple comparisons on chart?", True)
    multiple_location = expand.radio("Multiple comparisons outside or inside the graph?", ("outside", "inside"))
    pvalue_text = expand.radio("Choose type of multiple comparison label", ("star", "simple", "full"))
    color_palette = expand.selectbox("Choose color palette",
                                     ("Set2", "Accent", "Blues", "BrBG", "Dark2", "GnBu", "Greys", "Oranges", "Paired",
                                      "Pastel1", "Purples", "Set1", "Set3", "Spectral", "Wistia", "autumn", "binary",
                                      "cividis", "cool",
                                      "coolwarm", "icefire", "inferno", "magma", "ocean", "plasma", "rainbow", "summer",
                                      "twilight", "viridis", "winter"))

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

    normality = pg.normality(df, dv=x_var, group=y_var,
                             method="normaltest" if normality_selected == "Omnibus test of normality" else "shapiro")

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
            res.tukey_hsd(df=df, res_var=x_var, xfac_var=y_var,
                          anova_model=f"{x_var}~C({y_var})+C({y_var2})+C({y_var}):C({y_var2})")
            st.write(res.tukey_summary)
            st.subheader("Multiple comparisons of the 2nd grouping variable")
            res.tukey_hsd(df=df, res_var=x_var, xfac_var=y_var2,
                          anova_model=f"{x_var}~C({y_var})+C({y_var2})+C({y_var}):C({y_var2})")
            st.write(res.tukey_summary)
            st.subheader("Multiple comparisons of interactions")
            res.tukey_hsd(df=df, res_var=x_var, xfac_var=[y_var, y_var2],
                          anova_model=f"{x_var}~C({y_var})+C({y_var2})+C({y_var}):C({y_var2})")
            st.write(res.tukey_summary)
            df_filtered = res.tukey_summary[res.tukey_summary['p-value'] < 0.05][['group1', 'group2']]
            tuples = [tuple(x) for x in df_filtered.to_numpy()]

            new_df = df.groupby([y_var, y_var2])[x_var].mean().reset_index()

            st.success("Interaction plot")
            fig = plt.figure(figsize=(12, 6))
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
    fig = plt.figure(figsize=(12, 6))
    error = None
    if error_selected == "95% confidence intervals":
        error = 95
    elif error_selected == "Standard error of the mean":
        error = 68
    else:
        error = "sd"

    if y_var2 == "None":
        ax = sns.barplot(x=y_var, y=x_var, data=df, ci=error, capsize=0.08 if cap_selected else 0,
                         palette=color_palette, errwidth=0.7)
    else:

        ax = sns.barplot(x=y_var, y=x_var,
                         hue=y_var2, data=df, ci=error,
                         capsize=0.08 if cap_selected else 0, palette=color_palette, errwidth=0.7)

    if y_var2 == "None":
        groups = len(df.groupby(y_var))
        widthbars = [bars_width] * groups
        for bar, newwidth in zip(ax.patches, widthbars):
            x = bar.get_x()
            width = bar.get_width()
            centre = x + width / 2.
            bar.set_x(centre - newwidth / 2.)
            bar.set_width(newwidth)

    if show_multiple and y_var2 == "None":
        ax, test_results = add_stat_annotation(ax, data=df, x=x_var, y=y_var, box_pairs=tuples, test='t-test_ind', text_format=pvalue_text, loc=multiple_location)
  #  if show_multiple and y_var2 != "None":
   #     ax, test_results = add_stat_annotation(ax, data=df, x=x_var, y=y_var, hue=y_var2, box_pairs=tuples,
     #                                          test='t-test_ind', text_format=pvalue_text, loc=multiple_location)

    st.pyplot(fig)

    fig = plt.figure(figsize=(12, 6))

    if y_var2 == "None":
        ax = sns.boxplot(x=y_var, y=x_var, data=df, width=.3, palette=color_palette)
        sns.swarmplot(x=y_var, y=x_var, data=df, size=3, color=".3", linewidth=0)
    else:
        ax = sns.boxplot(x=y_var, y=x_var, hue=y_var2, data=df, palette=color_palette)

    if show_multiple and y_var2 == "None":
        ax, test_results = add_stat_annotation(ax, data=df, x=x_var, y=y_var, box_pairs=tuples,
                                               test='t-test_ind', text_format=pvalue_text, loc=multiple_location)
    if show_multiple and y_var2 != "None":
        ax, test_results = add_stat_annotation(ax, data=df, x=x_var, y=y_var, hue=y_var2, box_pairs=tuples,
                                               test='t-test_ind', text_format=pvalue_text, loc=multiple_location)

    st.pyplot(fig)

if choose_analysis == "Non-parametric ANOVA":

    numeric_vars = df.select_dtypes(include=np.number).columns.tolist()
    cat_vars = df.select_dtypes(include=np.object).columns.tolist()

    x_var = st.sidebar.selectbox("Choose numeric variable:", numeric_vars)
    y_var = st.sidebar.selectbox("Choose grouping variable:", cat_vars)

    expand = st.sidebar.beta_expander("More options")

    error_selected = expand.radio("Choose error bar to plot",
                                  ["Standard error of the mean", "95% confidence intervals", "Standard deviation"])
    bars_width = expand.slider("Choose bar width", 0.0, 1.5, 0.5, 0.05)
    cap_selected = expand.checkbox("Plot caps on error bars")
    show_multiple = expand.checkbox("Show multiple comparisons on chart?", True)
    multiple_location = expand.radio("Multiple comparisons outside or inside the graph?", ("outside", "inside"))
    pvalue_text = expand.radio("Choose type of multiple comparison label", ("star", "simple", "full"))
    color_palette = expand.selectbox("Choose color palette",
                                     ("Set2", "Accent", "Blues", "BrBG", "Dark2", "GnBu", "Greys", "Oranges", "Paired",
                                      "Pastel1", "Purples", "Set1", "Set3", "Spectral", "Wistia", "autumn", "binary",
                                      "cividis", "cool",
                                      "coolwarm", "icefire", "inferno", "magma", "ocean", "plasma", "rainbow", "summer",
                                      "twilight", "viridis", "winter"))

    st.header("Difference in means between groups results")
    st.success("Descriptive statistics are being calculated")
    function_dict = {x_var: ["mean", "std", "sem", "count"]}

    new = pd.DataFrame(df.groupby(y_var).aggregate(function_dict))
    st.write(new)

    results = pg.kruskal(data=df, dv=x_var, between=y_var, detailed=True)
    st.success("Kruskal-Wallis non-parametric ANOVA results")
    st.write(results)

    st.success("Games-Howell multiple comparisons")
    games_howell = pg.pairwise_gameshowell(dv=x_var, between=y_var, data=df).round(3)
    st.write(games_howell)
    df_filtered = games_howell[games_howell['pval'] < 0.05][['A', 'B']]
    tuples = [tuple(x) for x in df_filtered.to_numpy()]

    st.markdown("## ")

    st.success("Bar plots with errors are being generated")
    fig = plt.figure(figsize=(12, 6))
    error = None
    if error_selected == "95% confidence intervals":
        error = 95
    elif error_selected == "Standard error of the mean":
        error = 68
    else:
        error = "sd"

    ax = sns.barplot(x=y_var, y=x_var, data=df, ci=error, capsize=0.08 if cap_selected else 0, palette=color_palette,
                     errwidth=0.7)

    groups = len(df.groupby(y_var))
    widthbars = [bars_width] * groups
    for bar, newwidth in zip(ax.patches, widthbars):
        x = bar.get_x()
        width = bar.get_width()
        centre = x + width / 2.
        bar.set_x(centre - newwidth / 2.)
        bar.set_width(newwidth)

    if show_multiple:
        ax, test_results = add_stat_annotation(ax, data=df, x=x_var, y=y_var, box_pairs=tuples,
                                               test='t-test_ind', text_format=pvalue_text, loc=multiple_location)

    st.pyplot(fig)

    fig = plt.figure(figsize=(12, 6))

    ax = sns.boxplot(x=y_var, y=x_var, data=df, width=.3, palette=color_palette)
    sns.swarmplot(x=y_var, y=x_var, data=df, size=3, color=".3", linewidth=0)

    if show_multiple:
        ax, test_results = add_stat_annotation(ax, data=df, x=x_var, y=y_var, box_pairs=tuples,
                                               test='t-test_ind', text_format=pvalue_text, loc=multiple_location)

    st.pyplot(fig)

if choose_analysis == "Repeated measures mixed ANOVA":

    numeric_vars = df.select_dtypes(include=np.number).columns.tolist()
    cat_vars = df.select_dtypes(include=np.object).columns.tolist()
    cat_vars2 = df.select_dtypes(include=np.object).columns.tolist()
    cat_vars2.insert(0, "None")

    subject_var = st.sidebar.selectbox("Choose subject", numeric_vars)
    x_var = st.sidebar.selectbox("Choose numeric dependant variable:", numeric_vars)
    y_var = st.sidebar.selectbox("Choose within-group factor variable:", cat_vars)
    y_var2 = st.sidebar.selectbox("Choose between-group factor variable:", cat_vars2)

    groups = list(df[y_var].unique())

    expand = st.sidebar.beta_expander("More options")
    error_selected = expand.radio("Choose error bar to plot",
                                  ["Standard error of the mean", "95% confidence intervals", "Standard deviation"])
    groups_selection = expand.multiselect("Choose the order of within factor", groups, groups)

    color_palette = expand.selectbox("Choose color palette",
                                     ("Set2", "Accent", "Blues", "BrBG", "Dark2", "GnBu", "Greys", "Oranges", "Paired",
                                      "Pastel1", "Purples", "Set1", "Set3", "Spectral", "Wistia", "autumn", "binary",
                                      "cividis", "cool",
                                      "coolwarm", "icefire", "inferno", "magma", "ocean", "plasma", "rainbow", "summer",
                                      "twilight", "viridis", "winter"))

    st.success("Descriptive statistics")

    error = None
    if error_selected == "95% confidence intervals":
        error = 95
    elif error_selected == "Standard error of the mean":
        error = 68
    else:
        error = "sd"

    if y_var2 == "None":
        st.write(df.groupby(y_var)[x_var].agg(['mean', 'std', 'sem']).round(2))
    else:
        st.write(df.groupby([y_var, y_var2])[x_var].agg(['mean', 'std', 'sem']).round(2))

    if y_var2 == "None":
        st.success("One-way repeated measures ANOVA results")
        st.write(pg.rm_anova(dv=x_var, within=y_var, subject=subject_var, data=df, detailed=True))
        st.success("Post-hoc tests results")
        st.write(pg.pairwise_ttests(dv=x_var, within=y_var, subject=subject_var, data=df))
        st.success("Plots are being generated")
        fig = plt.figure(figsize=(12, 6))

        try:
            ax = sns.pointplot(data=df, x=y_var, y=x_var, capsize=.06, errwidth=0.7, ci=error, order=groups_selection)
            st.pyplot(fig)
        except:
            st.error("Please specify at least one within level!")

        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        df = df.query("{0} == @groups_selection".format(y_var))

        try:
            ax = pg.plot_paired(data=df, dv=x_var, within=y_var,
                                subject=subject_var, boxplot_in_front=True, ax=ax1, order=groups_selection)
            st.pyplot(fig)
        except:
            st.error("Please specify at least one within level!")

    else:
        st.success("Repeated measures mixed ANOVA results")
        st.write(pg.mixed_anova(dv=x_var, within=y_var, subject=subject_var, between=y_var2, data=df))
        st.success("Post-hoc tests results")
        st.write(pg.pairwise_ttests(dv=x_var, within=y_var, subject=subject_var, between=y_var2, data=df))
        st.success("Plots are being generated")
        fig = plt.figure(figsize=(12, 6))

        try:
            ax = sns.pointplot(data=df, x=y_var, y=x_var, hue=y_var2, dodge=True, markers=['o', 's'],
                               capsize=.06, errwidth=0.7, palette=color_palette, ci=error, order=groups_selection)
            st.pyplot(fig)
        except:
            st.error("Please specify at least one within level!")

if choose_analysis == "Correlation":

    numeric_vars = df.select_dtypes(include=np.number).columns.tolist()

    x_var1 = st.sidebar.selectbox("Choose first numeric variable", numeric_vars)
    x_var2 = st.sidebar.selectbox("Choose second numeric variable", numeric_vars)

    expand = st.sidebar.beta_expander("More options")
    method_selected = expand.radio("Select correlation type",
                                   ("pearson", "spearman", "kendall", "bicor", "percbend", "shepherd"))
    plot_r = expand.checkbox("Plot coefficient and p-value on graph?", True)
    font_scale = expand.slider("Font scale", 0.0, 4.0, 1.0, 0.1)

    start_x = float(df[x_var1].max() / 10)
    start_y = float(df[x_var2].max() - df[x_var2].max() / 10)
    max_x = float(df[x_var1].max())
    max_y = float(df[x_var2].max())
    x_pos = expand.slider("X position for the label", 0.0, max_x, start_x, (max_x / 100 + 0.1))
    y_pos = expand.slider("Y position for the label", 0.0, max_y, start_y, (max_y / 100 + 0.1))

    sns.set(style='white', font_scale=font_scale)

    st.success("Correlation results")

    corr_result = pg.corr(x=df[x_var1], y=df[x_var2], method=method_selected)
    st.write(corr_result)

    st.success("Correlation matrices")

    st.write(pg.pairwise_corr(df, padjust='bonf', method=method_selected).sort_values(by=['p-unc']))

    st.write(df.rcorr(padjust='bonf'))

    st.success("Correlation plot with distributions is being generated")
    fig = plt.figure(figsize=(12, 6))
    g = sns.JointGrid(data=df, x=x_var1, y=x_var2, height=6)
    g = g.plot_joint(sns.regplot, color="xkcd:muted blue")
    g = g.plot_marginals(sns.distplot, kde=False, bins=12, color="xkcd:bluey grey")
    if plot_r:
        g.ax_joint.text(x_pos, y_pos,
                        f"r = {corr_result.iloc[0].loc['r'].round(3)}, p = {corr_result.iloc[0].loc['p-val'].round(3)}",
                        fontstyle='italic')

    st.pyplot(g)

    corrs = df.corr(method=method_selected if method_selected in ['pearson', 'spearman', 'kendall'] else 'pearson')
    mask = np.zeros_like(corrs)
    mask[np.triu_indices_from(mask)] = True
    fig = plt.figure(figsize=(12, 6))
    ax = sns.heatmap(corrs, cmap='Spectral_r', mask=mask, square=True, vmin=-1, vmax=1, annot=True, linewidth=0.3)
    st.pyplot(fig)
