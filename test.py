import streamlit as st

placeholder = st.empty()
placeholder_slider = st.empty()

input = placeholder.text_input('text')
click_clear = placeholder_slider.slider('clear text input')
if click_clear:
    input = placeholder.text_input('text', value=str(click_clear), key=1)
if input:
    click_clear = placeholder_slider.slider('clear text input', value=float(input), key=1)
