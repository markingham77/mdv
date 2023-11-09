# mdv = Multi Dimensional View
import datetime
import streamlit as st
import os
import pandas as pd
import numpy as np
import altair as alt
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from dateutil.parser import parse

st.set_page_config(page_title="Multi-Dimensional Data Viewer",layout="wide")

def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False
    
st.title('Multi-Dimension Data Viewer')

file_uploaded = st.file_uploader('Upload your csv')
categorical_columns=[]
quantitative_columns=[]
date_columns=[]
other_columns=[]
if file_uploaded != None:
    @st.cache_data
    def load_data(file_uploaded):
        """
        loads data via 'teimseries.sql' tempalte and tben expands the comma-delimited
        lists of splits and split_values (one coloumn for each split)
        """
        return pd.read_csv(file_uploaded, keep_default_na=False, na_values="")   

    data = load_data(file_uploaded)
    for column in data.columns:
        if is_categorical_dtype(data[column]) or data[column].nunique() < 10:
            categorical_columns.append(column)
        elif is_numeric_dtype(data[column]):
            quantitative_columns.append(column)
        elif is_datetime64_any_dtype(data[column]):
            date_columns.append(column)
        else:
            col_data = data[column][data[column]!='NaN']
            col_data = col_data[pd.notna(col_data)]
            if type(col_data.iloc[0])==str:
                if is_date(col_data.iloc[0]):
                    date_columns.append(column)
                else:
                    other_columns.append(column)
            else:
                if type(col_data.iloc[0])==datetime.date:
                    date_columns.append(column)
                else:
                    other_columns.append(column)   

    splits = categorical_columns
    

    def filter_dataframe(df: pd.DataFrame, tab, columns=[]) -> pd.DataFrame:
        """
        Adds a UI on top of a dataframe to let viewers filter columns

        Args:
            df (pd.DataFrame): Original dataframe

        Returns:
            pd.DataFrame: Filtered dataframe

        source: https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/#bringing-it-all-together    
        """
        df = df.copy()
        if len(columns)>0:
            df=df[columns]

        modify = tab.checkbox("Add filters")
        if not modify:
            return df

        # Try to convert datetimes into a standard format (datetime, no timezone)
        for col in df.columns:
            if is_object_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

            if is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)

        modification_container = tab.container()

        with modification_container:
            to_filter_columns = tab.multiselect("Filter dataframe on", df.columns)
            for column in to_filter_columns:
                left, right = tab.columns((1, 20))
                # Treat columns with < 10 unique values as categorical
                if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        default=list(df[column].unique()),
                    )
                    df = df[df[column].isin(user_cat_input)]
                elif is_numeric_dtype(df[column]):
                    _min = float(df[column].min())
                    _max = float(df[column].max())
                    step = (_max - _min) / 100
                    user_num_input = right.slider(
                        f"Values for {column}",
                        min_value=_min,
                        max_value=_max,
                        value=(_min, _max),
                        step=step,
                    )
                    df = df[df[column].between(*user_num_input)]
                elif is_datetime64_any_dtype(df[column]):
                    user_date_input = right.date_input(
                        f"Values for {column}",
                        value=(
                            df[column].min(),
                            df[column].max(),
                        ),
                    )
                    if len(user_date_input) == 2:
                        user_date_input = tuple(map(pd.to_datetime, user_date_input))
                        start_date, end_date = user_date_input
                        df = df.loc[df[column].between(start_date, end_date)]
                else:
                    user_text_input = right.text_input(
                        f"Substring or regex in {column}",
                    )
                    if user_text_input:
                        df = df[df[column].astype(str).str.contains(user_text_input)]
        return df


    def rot(x):
        """
        roatates a list by 1
        """
        return x[1:] + x[:1]

    
    categorical = date_columns + [x.upper() for x in splits]
    quantitative = quantitative_columns

    x_item = st.sidebar.selectbox(
        "X-Axis Item",
        date_columns + quantitative
    )
    remaining_categories = [x for x in categorical if x != x_item]
    row = st.sidebar.selectbox(
        "Rows",
        remaining_categories
    )
    row_ex_nan = st.sidebar.checkbox(f'exlude NaNs',key='row')
    if row_ex_nan:
        data = data[~data[row].isna()]
        data = data[data[column]!='NaN']
        data = data[data[row]!='nan']

    remaining_categories=rot(remaining_categories)
    column = st.sidebar.selectbox(
        "Columns",
        remaining_categories
    )
    column_ex_nan = st.sidebar.checkbox(f'exlude NaNs',key='column')
    if column_ex_nan:
        data = data[~data[column].isna()]
        data = data[data[column]!='NaN']
        data = data[data[column]!='nan']
        
    remaining_categories=rot(remaining_categories)
    color = st.sidebar.selectbox(
        "Colour",
        remaining_categories
    )
    color_ex_nan = st.sidebar.checkbox(f'exlude NaNs',key='colour')
    if color_ex_nan:
        data = data[~data[color].isna()]
        data = data[data[column]!='NaN']
        data = data[data[color]!='nan']

    remaining_categories=rot(remaining_categories)

    remaining_dims = [x for x in categorical if x not in [x_item, color, row,column]]
    remaining_dim_values=[]
    for dim in remaining_dims:
        x = st.sidebar.selectbox(
            f'{dim}',
            [x for x in data[dim].unique()] + ['All']    
        )
        if x!='All':
            data = data[data[dim]==x]
        remaining_dim_values.append(x)


    tabs = st.tabs(quantitative + ['DATA'])
    # the actual plots
    for i,tab in enumerate(tabs):
        if i>=len(quantitative):
            # then at end of tabs and must be "Data" tab, so produce dataframe
            columns = categorical+quantitative
            df = filter_dataframe(data,tab,columns)
            tab.dataframe(df)
            summary = tab.checkbox("Summary")
            if summary:
                tab.dataframe(df.describe())
            # tab.dataframe(data)
        else:
            y_item=quantitative[i]
            chart_type = tab.selectbox('Chart Type',['line', 'stacked bar', 'scatter'],key=y_item)
            if chart_type == 'stacked bar':
                chart = alt.Chart(data).mark_bar()
            elif chart_type == 'line':
                chart = alt.Chart(data).mark_line()
            elif chart_type == 'scatter':
                chart = alt.Chart(data).mark_circle()                
            c = chart.encode(
                alt.X(x_item + ':T',axis=alt.Axis(tickSize=0, labelFontSize=0, grid=False)).title(''),
                alt.Y(f'{y_item}:Q').title(''),
                color=color,
                tooltip = [x_item,y_item,color]
            ).properties(
                width=80,
                height=60
            ).facet(
                column=f'{column}:N',
                row=f'{row}:N',
                # title=facet_expander
            )
            tab.subheader(y_item)
            tab.altair_chart(c, use_container_width=True)






        