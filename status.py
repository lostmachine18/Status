import streamlit as st
import pandas as pd
import xlrd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

global df
describe_data = None

st.title("Status")
st.markdown("## _Your personal status of data analysis_")
st.markdown("____")

expand = st.sidebar.beta_expander("See more")
expand.checkbox("Cool")

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
    choose_analysis = st.sidebar.selectbox("Choose analysis to perform:", ("None", "Chi-square test", "Student t-test", "One-way ANOVA"))

if describe_data:
    pr = ProfileReport(df, explorative=True)
    st_profile_report(pr)



