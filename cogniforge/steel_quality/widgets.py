from collections.abc import Iterable
from typing import Generic, TypeVar

import streamlit as st
from furthrmind import collection

from cogniforge.utils.furthr import get_furthr_client, is_fielddata_match

T = TypeVar('T', collection.Group, collection.Experiment, collection.Sample, collection.ResearchItem)
Z = TypeVar('Z')


class FurthrCollectionWrapper(Generic[T]):
    def __init__(self, raw: T, file_extension: str | None) -> None:
        self.raw = raw
        self.file_extension = file_extension # maybe use for download?
        self.id = getattr(raw, 'id', raw._id)

        for field in raw.fielddata:
            # Python attribute names should be snake_case
            name = field.field_name.lower().replace(' ', '_')
            value = FurthrCollectionWrapper.__clean_field_value(field)
            # Make metadata easily available
            setattr(self, name, value)

    # support for Streamlit cache_data
    def __reduce__(self) -> tuple[str]:
        # Without the comma, would return str instead of tuple[str]. See:
        # https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences
        return (self.id, )

    @staticmethod
    def __clean_field_value(field: collection.FieldData):
        if field.field_type == 'CheckBox':
            # Otherwise would return None instead of False,
            # which is bad for display purposes.
            return bool(field.value)
        if field.field_type == 'Numeric':
            # Currently, decimal places are never used
            # in metadata. Cast for shorter display.
            return int(field.value)
        return field.value


def _test_container_match(
        container_fielddata: dict[str, str | int | bool] | None,
        file_extension: str | None
):
    def fn(candidate) -> bool:
        candidate.get() # otherwise metadata empty
        return (
            (not container_fielddata or is_fielddata_match(candidate.fielddata, container_fielddata)) and
            (not file_extension or any(f.name.endswith(file_extension) for f in candidate.files))
        )

    return fn


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
        return FurthrCollectionWrapper(group, file_extension)
    
    if collection_type is collection.Experiment:
        label = "Choose an experiment"
        containers = group.experiments
    elif collection_type is collection.Sample:
        label = "Choose a sample"
        containers = group.samples
    elif collection_type is collection.ResearchItem and collection_category:
        label = f"Choose a {collection_category} item"
        containers = group.researchitems.get(collection_category, [])

    if container_fielddata or file_extension:
        containers = filter(_test_container_match(container_fielddata, file_extension), containers)    

    container = select(label, containers, "item")

    if container:
        return FurthrCollectionWrapper(container, file_extension)

    return None


def resolution(collection: FurthrCollectionWrapper) -> None:
    if collection:
        st.info(f"**Resolution:** {collection.image_width}x{collection.image_height} px")


@st.fragment # does this improve anything?
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