#app.py
import streamlit as st
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from st_aggrid import AgGrid, GridOptionsBuilder

from clearml_get import get_by_project_name, get_df_by_project_id


def display_data(data):

    gb = GridOptionsBuilder()

    # makes columns resizable, sortable and filterable by default
    gb.configure_default_column(
        resizable=True,
        filterable=True,
        sortable=True,
        editable=False,
        groupable=True,
    )

    for col in data.columns:
        hide = data[col].nunique() == 1
        pinned = col in st.session_state.pinned
        if is_numeric_dtype(data[col]):
            gb.configure_column(field=col, header_name=col, editable=False, type=["numericColumn"], aggFunc="max",hide=hide, pinned=pinned)
        else:
            gb.configure_column(field=col, header_name=col, editable=False, hide=hide, pinned=pinned)

    #makes tooltip appear instantly
    gb.configure_grid_options(tooltipShowDelay=0)
    go = gb.build()

    AgGrid(data, gridOptions=go)
    
@st.cache
def load_data(project_ids):
    data = get_df_by_project_id(project_ids)
    return data


st.set_page_config(layout="wide")

filter_test = st.checkbox("Filter Test", value=True)
project_name = st.text_input("Project Name", "example*")
projects = get_by_project_name(project_name)
projects = [project for project in projects if "test" not in project.name] if filter_test else projects

project_names = [project.name for project in projects]
project_dict = {project.name: project.id for project in projects}
selected_projects = st.multiselect("Select Project", project_names)

if "selected_projects" not in st.session_state:
    st.session_state.selected_projects = set()

st.session_state.selected_projects = st.session_state.selected_projects.union(set(selected_projects))

st.write(st.session_state.selected_projects)

if st.button("Clear Data"):
    st.session_state.selected_projects = set()
    
st.session_state.pinned = st.multiselect("Pinned", ['in_params/experiment/group', 'f1', 'Group'])

if st.checkbox("Load Data", key="load_data"):
    data = load_data([project_dict[project] for project in selected_projects])
    display_data(data)


