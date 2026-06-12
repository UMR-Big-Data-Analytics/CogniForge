import itertools
import os
from collections.abc import Callable
from io import BytesIO, StringIO
from tempfile import NamedTemporaryFile
from typing import Any, Generic, TypeVar

import config
import pandas as pd
import requests
import streamlit as st
import tensorflow as tf
from furthrmind import Furthrmind as API
from furthrmind.collection import Experiment, FieldData, File, Group, ResearchItem, Sample
from furthrmind.collection.baseclass import BaseClass
from furthrmind.file_loader import FileLoader
from matplotlib.figure import Figure

from . import http_cache
from .object_select_box import selectbox
from .state_button import button

C = TypeVar('C', Group, Experiment, Sample, ResearchItem)
C2 = TypeVar('C2', Group, Experiment, Sample, ResearchItem)


class CollectionWrapper(Generic[C]):
    def __init__(self, raw: C, file_extension: str | None = None) -> None:
        # fetch fielddata if not already loaded
        if not raw._fetched:
            raw.get()

        self.raw = raw
        self.file_extension = file_extension
        self.id = getattr(raw, 'id', raw._id)
    
    def __str__(self):
        if isinstance(self.raw, ResearchItem):  # noqa: SIM108
            kind = self.raw.category.name
        else:
            kind = type(self.raw).__name__
            
        return f"{kind} '{self.raw.name}' (ID: {self.id})"
    
    @staticmethod
    def __to_furthr_field_name(python_attribute_name: str) -> str:
        # Python attribute names are snake_case.
        # The database uses "Title Case".
        return ' '.join(o.capitalize() for o in python_attribute_name.split('_'))
    
    # Make metadata easily available
    def __getattr__(self, name: str):
        name = CollectionWrapper.__to_furthr_field_name(name)

        try:
            field = next(o for o in self.raw.fielddata if o.field_name == name)

            if field.field_type == 'CheckBox':
                # Otherwise would return None instead of False,
                # which is bad for display purposes.
                return bool(field.value)
            
            if field.field_type == 'Numeric':
                # Currently, decimal places are never used
                # in metadata. Cast for shorter display.
                return int(field.value)
            
            return field.value
        except StopIteration as ex:
            raise AttributeError(f"The metadata does not contain a field called '{name}'") from ex    
    
    # Easily update metadata
    def __setattr__(self, name: str, value):
        if name in ['raw', 'file_extension', 'id']:
            super().__setattr__(name, value)
            return
        
        name = CollectionWrapper.__to_furthr_field_name(name)
        
        if any(o for o in self.raw.fielddata if o.field_name == name):
            self.raw.update_field_value(value, field_name=name)
            return
        
        if isinstance(value, bool):
            field_type = 'CheckBox'
        elif isinstance(value, (int, float)):
            field_type = 'Numeric'
        else:
            field_type = 'SingleLine'

        self.raw.add_field(field_name=name, field_type=field_type, value=value)

    # support for Streamlit cache_data
    def __reduce__(self) -> tuple[str]:
        # Without the comma, would return str instead of tuple[str]. See:
        # https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences
        return (self.id, )
    
    @staticmethod
    def __make_serializer(content) -> Callable[[str], None]:
        if isinstance(content, pd.DataFrame):
            def serializer(file_path: str):
                content.to_csv(file_path, index=False)
            return serializer
        
        if isinstance(content, tf.keras.Model):
            return content.save
        
        if isinstance(content, Figure):
            return content.savefig

        raise TypeError("Expected a DataFrame/Model/Figure, got " + type(content).__name__)
    
    def __throw_if_files_unsupported(self):
        if not isinstance(self.raw, (Experiment, Sample, ResearchItem)):
            raise TypeError("Operation supported on Experiment/Sample/ResearchItem, but self is " + self)

    def download_files(self) -> list[tuple[BytesIO, str]] | None:
        self.__throw_if_files_unsupported()
        return download_item_bytes(self.raw, self.file_extension)
    
    def upload_file(self, file_path: str, target_name: str | None = None):
        self.__throw_if_files_unsupported()
        self.raw.add_file(file_path, target_name)
    
    def upload_content(self, content: pd.DataFrame | tf.keras.Model | Figure, target_name: str):
        self.__throw_if_files_unsupported()

        # Throw early on unexpected content type
        serializer = CollectionWrapper.__make_serializer(content)

        target_extension = os.path.splitext(target_name)[1]

        if not target_extension:
            raise ValueError("target_name must contain the desired file extension")
        
        # See https://stackoverflow.com/a/8577226
        fh = NamedTemporaryFile(delete=False, suffix=target_extension)  # noqa: SIM115

        try:
            serializer(fh.name)
            self.raw.add_file(fh.name, target_name)
        finally:
            fh.close()
            os.remove(fh.name)
    
    def add_link_to(self, target: 'CollectionWrapper'):
        self.__throw_if_files_unsupported()

        if isinstance(target.raw, Experiment):
            self.raw.add_linked_experiment(target.id)
        elif isinstance(target.raw, Sample):
            self.raw.add_linked_sample(target.id)
        elif isinstance(target.raw, ResearchItem):
            self.raw.add_linked_researchitem(target.id)
        else:
            raise TypeError("Expected an Experiment/Sample/ResearchItem wrapped in CollectionWrapper")
    
    def sub_collection(self, name: str, kind: type[C2], category: str | None = None) -> 'CollectionPlaceholder[C2]':
        if not isinstance(self.raw, Group):
            raise TypeError("Operation supported on Group, but self is " + self)
        
        return CollectionPlaceholder(name, self.raw, kind, category)


class CollectionPlaceholder(Generic[C]):
    def __init__(
            self,
            name: str,
            parent: Group,
            kind: type[C],
            category: str | None
    ):
        if kind is ResearchItem and not category:
            raise ValueError("A category must be specified in case of ResearchItem")
        
        if category and kind is not ResearchItem:
            raise ValueError("A category can only be set for ResearchItems")

        self.name = name
        self.parent = parent
        self.kind = kind
        self.category = category
    
    def __str__(self):
        return f"{self.category or self.kind.__name__} '{self.name}'"
    
    @property
    def siblings(self) -> list[C]:
        # fetch siblings if not already loaded
        if not self.parent._fetched:
            self.parent.get()

        if self.kind is Experiment:
            return self.parent.experiments
        elif self.kind is Sample:
            return self.parent.samples
        elif self.kind is ResearchItem:
            return self.parent.researchitems.get(self.category, [])
        elif self.kind is Group:
            return self.parent.sub_groups

    @property
    def exists(self) -> bool:
        return any(x.name == self.name for x in self.siblings)
    
    def create(self) -> CollectionWrapper[C]:
        # without this a too generic error would be thrown
        if self.exists:
            raise ValueError(f"Cannot create {self} in parent group '{self.parent.name}' because it already exists")  # noqa: E501
        
        if self.kind is ResearchItem:
            instance = ResearchItem.create(self.name, group_id=self.parent.id, category_name=self.category)
        elif self.kind is Group:
            raise NotImplementedError("BROKEN: Group creation not possible. FURTHRmind differs in parameter names and types regarding between Python-SDK, serverside input validation and parameter parsing.")
            #instance = Group.create(self.name, parent_group_id=self.parent.id)
        else:
            # Sample or Experiment
            instance = self.kind.create(self.name, group_id=self.parent.id)

        return CollectionWrapper(instance)
    
    def get(self) -> CollectionWrapper[C]:
        try:
            instance = next(x.name == self.name for x in self.siblings)
            return CollectionWrapper(instance)
        except StopIteration as ex:
            raise ValueError(f"Did not find {self} in parent group '{self.parent.name}'") from ex


@st.cache_resource
def get_furthr_client():
    session = requests.Session()
    session.headers.update({
        "X-API-KEY": config.furthr['api_key'],
        "Content-Type": "application/json"
    })
    fm = API(
        host=config.furthr['host'],
        api_key=config.furthr['api_key'],
        project_id=config.furthr['project_id']
    )
    return fm, session


def __download_item(
    item: Experiment | Sample | ResearchItem | File | None,
    file_extension: str | None,
    download_fn: Callable[[str, dict], BytesIO | StringIO | requests.Response]
) -> tuple[BytesIO, str] | tuple[StringIO, str] | None:
    if item is None:
        return None
    
    if isinstance(item, Group):
        st.error("Cannot download group")
        return None
    
    _, session = get_furthr_client()
    
    if isinstance(item, File):
        b = download_fn(f"{config.furthr['host']}/files/{item.id}", dict(session.headers))
        if isinstance(b, (BytesIO, StringIO)):
            return (b, item.name)
        else:
            st.error(f"Failed to download file {item.name}: {b.reason}")
            return None
    
    # x must be a container when reaching this point
    if not item._fetched:
        item.get()

    files = item.files

    if file_extension:
        files = [f for f in files if f.name.endswith(file_extension)]

    if not files:
        if file_extension:
            st.error(f"No {file_extension} files found")
        else:
            st.error("No files found")
        return None
    
    if len(files) == 1:
        # shortcut progress bar
        return [__download_item(files[0], file_extension, download_fn)]

    downloaded_files = []
    bar_title = "Download in progress. Please wait."
    my_bar = st.progress(0, text=bar_title)

    for index, file in enumerate(files):
        downloaded_files.append(__download_item(file, file_extension, download_fn))
        my_bar.progress(index / len(files), text=bar_title)

    my_bar.empty()
    return downloaded_files


def download_item_string(
    item: Experiment | Sample | ResearchItem | File | None,
    file_extension: str | None = None
 ) -> tuple[StringIO, str] | None: 
    return __download_item(item, file_extension, http_cache.get_as_string)


def download_item_bytes(
    item: Experiment | Sample | ResearchItem | File | None,
    file_extension: str | None = None
 ) -> tuple[BytesIO, str] | None: 
    return __download_item(item, file_extension, http_cache.get_as_bytes)


def hash_furthr_item(item: BaseClass) -> str:
    return getattr(item, 'id', item._id)


def is_fielddata_match(
        found_fielddata: FieldData,
        expected_fielddata: dict[str, str],
) -> bool:
    for field_name, expected_value in expected_fielddata.items():
        try:
            found_field = next(o for o in found_fielddata if o.field_name == field_name)
        except StopIteration:
            return False
        
        if expected_value == "ANY":
            continue
        
        if found_field.field_type == "ComboBox":  # noqa: SIM108
            found_value = found_field.value['name']
        else:
            found_value = found_field.value

        if found_value != expected_value:
            return False

    return True


# Use furthr_selectbox instead
class FURTHRmind:
    """FURTHRmind API interface."""

    __id: str
    fm: API
    __selected: Group | Experiment | Sample | ResearchItem | File | None = None
    force_group_id: str | None = None
    file_extension: str | None = None
    container_category: str | None = None
    expected_fielddata: dict[str, str] | None = None

    @property
    def selected(self) -> Group | Experiment | Sample | ResearchItem | File | None:
        """Return currently selected group/experiment/sample/research item/file"""
        return self.__selected

    def __init__(self, id: str = "furtherwidget"):
        """Setup the project for the API."""
        self.__id = id
        self.fm, _ = get_furthr_client()
    
    def select_group(self):
        """Select a group from the project."""
        if self.force_group_id is not None:
            self.__selected = self.fm.Group.get(id=self.force_group_id)
        else:
            groups = self.fm.Group.get_all()
            self.__selected = selectbox(groups, label="Choose a group", key=f"{self.__id}_group")

        if self.__selected is None:
            st.error("No group found")

    def select_container(self):
        """Select a research item, experiment or sample from the group which has a specific attribute and field data"""
        self.select_group()
        group: Group | None = self.__selected
        if group is None:
            return
        
        group.get()

        if self.container_category == "experiment":
            label = "Choose an experiment"
            items = group.experiments
        elif self.container_category == "sample":
            label = "Choose a sample"
            items = group.samples
        elif self.container_category == "researchitem":
            label = "Choose a research item"
            items = group.group.researchitems.values()
        elif self.container_category:
            label = f"Choose a {self.container_category} item"
            items = group.researchitems.get(self.container_category, [])
        else:
            label = "Choose an experiment/sample/research item"
            items = group.experiments + group.samples
            items.extend(itertools.chain.from_iterable(group.researchitems.values()))

        if self.expected_fielddata:
            # fielddata not populated without get
            items = [
                o for o in items
                if is_fielddata_match(o.get().fielddata, self.expected_fielddata)
            ]
        
        self.__selected = selectbox(items, label=label, key=f"{self.__id}_item")

        if self.__selected is None:
            st.error("No container found")

    def select_file(self):
        """Select a file from the experiment or sample."""
        self.select_container()
        container: Experiment | ResearchItem | Sample | None = self.__selected
        if container is None:
            return
        
        if not container._fetched:
            container.get()

        files: list[File] = container.files

        if self.file_extension:
            files = [f for f in files if f.name.endswith(self.file_extension)]
            label = f"Choose a {self.file_extension} file"
        else:
            label = "Choose a file"

        self.__selected = selectbox(
            files, label=label, key=f"{self.__id}_file"
        )

        if self.__selected is None:
            if self.file_extension:
                st.error(f"No {self.file_extension} file found")
            else:
                st.error("No file found")

    def download_bytes_button(self) -> tuple[BytesIO, str] | None:
        """Download the selected item as bytes from the FURTHRmind database."""
        if self.__selected and button("Load", "load" + self.__id, stateful=True):
            return download_item_bytes(self.__selected, self.file_extension)
        
    def download_string_button(self) -> tuple[BytesIO, str] | None:
        """Download the selected item as string from the FURTHRmind database."""
        if self.__selected and button("Load", "load" + self.__id, stateful=True):
            return download_item_string(self.__selected, self.file_extension)
    
    def upload_file(self, path_or_writer: str | Callable[[str], tuple[Any, str]]) -> None:
        """Upload a file to the FURTHRmind database."""
        self.select_container()
        container: Experiment | Sample | ResearchItem | None = self.__selected

        if container is None or not st.button("Upload", "upload" + self.__id):
            return
        
        if isinstance(container, Experiment):
            container_type = "experiment"
        elif isinstance(container, Sample):
            container_type = "sample"
        else:
            container_type = "researchitem"

        container_description = {
            "project": config.furthr['project_id'],
            "type": container_type,
            "id": container.id,
        }
        file_loader = FileLoader(config.furthr['host'], config.furthr['api_key'])

        if isinstance(path_or_writer, str):
            file_loader.uploadFile(filePath=path_or_writer, parent=container_description)
        else:
            # useful to be able to only create a temp file when upload was pressed
            with tempfile.NamedTemporaryFile(delete_on_close=False) as fh:
                _, file_name = path_or_writer(fh.name)
                file_loader.uploadFile(
                    filePath=fh.name,
                    fileName=file_name,
                    parent=container_description
                )
                st.success("Anomaly score uploaded to furthrmind database")

    def upload_csv(self, csv: pd.DataFrame, name: str) -> None:
        """Upload a CSV file to the FURTHRmind database."""
        self.upload_file(lambda tmp_path: (
            csv.to_csv(tmp_path, index=False),
            name + ".csv"
        ))
