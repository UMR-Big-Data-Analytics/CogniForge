from collections.abc import Iterable
from datetime import datetime

import streamlit as st
from furthrmind import collection

from cogniforge.utils import furthr


def _test_container_match(
        container_fielddata: dict[str, str | int | bool] | None,
        file_extension: str | None
):
    def fn(candidate) -> bool:
        candidate.get() # otherwise metadata empty
        return (
            (not container_fielddata or furthr.is_fielddata_match(candidate.fielddata, container_fielddata)) and
            (not file_extension or any(f.name.endswith(file_extension) for f in candidate.files))
        )

    return fn


def furthr_open_collection(
        key: str,
        kind: type[furthr.C],
        category: str | None = None,
        container_fielddata: dict[str, str | int | bool] | None = None,
        force_group_id: str | None = None,
        file_extension: str | None = None
) -> furthr.CollectionWrapper[furthr.C] | None:
    def get_option_name(option) -> str:
        return option.name

    def select(label: str, options: list[furthr.C2], subkey: str) -> furthr.C2 | None:
        res = st.selectbox(label, options, None, get_option_name, f"{key}_{subkey}")
        return res.get() if res else None

    if kind is collection.ResearchItem and not category:
        raise ValueError("A collection_category must be specified in case of ResearchItem")

    fm, _ = furthr.get_furthr_client()

    if force_group_id:
        group = fm.Group.get(force_group_id)
    else:
        groups = fm.Group.get_all()
        group = select("Choose a group", groups, "group")

    if not group:
        return None
    
    if kind is collection.Group:
        return furthr.CollectionWrapper(group, file_extension)
    
    if kind is collection.Experiment:
        label = "Choose an experiment"
        containers = group.experiments
    elif kind is collection.Sample:
        label = "Choose a sample"
        containers = group.samples
    elif kind is collection.ResearchItem:
        label = f"Choose a {category} item"
        containers = group.researchitems.get(category, [])

    if container_fielddata or file_extension:
        containers = filter(_test_container_match(container_fielddata, file_extension), containers)    

    container = select(label, containers, "item")

    if container:
        return furthr.CollectionWrapper(container, file_extension)

    return None


def resolution(collection: furthr.CollectionWrapper) -> None:
    if collection:
        st.info(f"**Resolution:** {collection.image_width}x{collection.image_height} px")


def form(
        key: str,
        inputs: dict[str, Iterable[str | bool]]
) -> dict[str, list[str | bool]]:
    results = {}
    ready = True

    for input_label, input_options in inputs.items():
        input_type = type(input_options[0])

        if input_type is bool:
            input_value = st.pills(input_label, input_options, selection_mode="multi", key=f"{key}_{input_label}")
        else:
            input_value = st.multiselect(input_label, input_options, key=f"{key}_{input_label}")

        ready = ready and len(input_value) > 0
        results[input_label] = input_value

    if ready:
        st.success("All necessary information was entered.")
        return results
    else:
        st.error("At least one value of every input must be selected.")
        return None
   

def furthr_save_collection(
        key: str,
        kind: type[furthr.C],
        category: str | None = None,
        overwrite_warning: bool = True
)-> furthr.CollectionPlaceholder[furthr.C] | None:
    if kind is collection.Experiment:
        kind_str = "experiment"
    elif kind is collection.Sample:
        kind_str = "sample"
    elif kind is collection.ResearchItem:
        kind_str = f"{category} item"
    else:
        raise TypeError("Expected a type of Experiment/Sample/ResearchItem")
    
    name = st.text_input("Type a name for the new " + kind_str, placeholder="Container Name", key=f"{key}_name")
    parent = furthr_open_collection(f"{key}_parent", collection.Group)

    if not (name and parent):
        return None
    
    placeholder = furthr.CollectionPlaceholder(name, parent.raw, kind, category)

    if overwrite_warning and placeholder.exists:
        st.warning(f"The group '{parent.raw.name}' already contains {kind_str} '{name}'")

    return placeholder


def log(message: str):
    st.markdown(f"`[{datetime.now().isoformat()}]` {message}")