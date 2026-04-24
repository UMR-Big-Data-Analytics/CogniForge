import itertools
from collections.abc import Iterable
from typing import Generic, TypeVar

import streamlit as st
from furthrmind import collection

from cogniforge.utils.furthr import get_furthr_client, is_fielddata_match

T = TypeVar('T', collection.Group, collection.Experiment, collection.Sample, collection.ResearchItem, collection.File)
Z = TypeVar('Z')


class FurthrCollectionWrapper(Generic[T]):
    def __init__(self, raw: T) -> None:
        self.raw = raw

        for field in raw.fielddata:
            # Python attribute names should be snake_case
            name = field.field_name.lower().replace(' ', '_')

            if field.field_type == 'CheckBox':
                # Otherwise would return None instead of False,
                # which is bad for display purposes.
                value = bool(field.value)
            elif field.field_type == 'Numeric':
                # Currently, decimal places are never used
                # in metadata. Cast for shorter display.
                value = int(field.value)
            else:
                value = field.value

            # Make metadata easily available
            setattr(self, name, value)


def furthr_selectbox(
        key: str,
        collection_type: type[T],
        collection_category: str | None = None,
        container_fielddata: dict[str, str | int | bool] | None = None,
        force_group_id: str | None = None,
        file_extension: str | None = None
) -> FurthrCollectionWrapper[T] | None:
    def get_option_name(option) -> str:
        return option.name

    def select(label: str, options: list[Z], subkey: str) -> Z | None:
        res = st.selectbox(label, options, None, get_option_name, f"{key}_{subkey}")
        return res.get() if res else None

    fm, _ = get_furthr_client()

    if force_group_id:
        group = fm.Group.get(force_group_id)
    else:
        groups = fm.Group.get_all()
        group = select("Choose a group", groups, "group")

    if not group:
        return None
    
    if collection_type is collection.Group:
        return FurthrCollectionWrapper(group)
    
    if collection_type is collection.Experiment:
        label = "Choose an experiment"
        items = group.experiments
    elif collection_type is collection.Sample:
        label = "Choose a sample"
        items = group.samples
    elif collection_type is collection.ResearchItem and collection_category:
        label = f"Choose a {collection_category} item"
        items = group.researchitems.get(collection_category, [])
    elif collection_type is collection.File:
        label = "Choose an experiment/sample/research item"
        items = group.experiments + group.samples
        items.extend(itertools.chain.from_iterable(group.researchitems.values()))

    if container_fielddata:
        items = [
            o for o in items
            if is_fielddata_match(o.get().fielddata, container_fielddata)
        ]
    
    container = select(label, items, "item")

    if not container:
        return None
    
    if collection_type is not collection.Field:
        return FurthrCollectionWrapper(container)
    
    files: list[collection.File] = container.files

    if file_extension:
        files = [f for f in files if f.name.endswith(file_extension)]
        label = f"Choose a {file_extension} file"
    else:
        label = "Choose a file"

    file = select(label, files, "file")

    if file:
        return FurthrCollectionWrapper(file)
    
    return None


def resolution(collection: FurthrCollectionWrapper) -> None:
    if collection:
        st.info(f"**Resolution:** {collection.image_width}x{collection.image_height} px")


@st.fragment
def form(
        key: str,
        inputs: dict[str, Iterable[str | bool]]
) -> list[list[str | bool]]:
    results = []
    ready = True

    for input_label, input_options in inputs.items():
        input_type = type(input_options[0])

        if input_type is bool:
            input_value = st.pills(input_label, input_options, selection_mode="multi", key=f"{key}_{input_label}")
        else:
            input_value = st.multiselect(input_label, input_options, key=f"{key}_{input_label}")

        ready = ready and len(input_value) > 0
        results.append(input_value)

    if ready:
        st.success("All necessary information was entered")
        return results
    else:
        st.error("At least one value of every input must be selected")
        return None