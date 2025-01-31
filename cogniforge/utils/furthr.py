import os
import tempfile
from enum import Enum
from io import BytesIO, StringIO
import itertools
from typing import Callable, Any

import pandas as pd
import requests
import streamlit as st
from furthrmind import Furthrmind as API
from furthrmind.collection import Experiment, File, Group, ResearchItem, Sample, FieldData
from furthrmind.file_loader import FileLoader
import config

from .object_select_box import selectbox
from .state_button import button


@st.cache_data
def api_get(url: str, headers: dict) -> dict:
    """Get data from the FURTHRmind API."""
    response = requests.get(url, headers=headers)
    return response.json()["results"]


@st.cache_data
def api_get_string(
    url: str, headers: dict, chunk_size: int = 8192000
) -> StringIO | requests.Response:
    """Get a CSV file from the FURTHRmind API."""
    response = requests.get(url, headers=headers)
    if response.ok:
        bytes = b""
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                bytes += chunk
        return StringIO(bytes.decode("latin"))
    else:
        return response


@st.cache_data
def api_get_bytes(
    url: str, headers: dict, chunk_size: int = 8192000
) -> BytesIO | requests.Response:
    """Get a CSV file from the FURTHRmind API."""
    response = requests.get(url, headers=headers)
    if response.ok:
        bytes_file = b""
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                bytes_file += chunk
        return BytesIO(bytes_file)
    else:
        return response


@st.cache_data
def _read_csv_spraying(csv: StringIO) -> pd.DataFrame:
    st.write(csv)
    return pd.read_csv(
        csv,
        sep=";",
        index_col=0,
        decimal=",",
        header=0,
        skip_blank_lines=True,
        skiprows=[1],
        converters={
            "Noise": lambda x: float(x.replace(",", ".")) if "," in x else 0.0,
            "Drahtrollendrehzahl": lambda x: (
                float(x.replace(",", ".")) if "," in x else 0.0
            ),
        },
        encoding="latin",
    )


@st.cache_data
def _read_csv_wire(csv: StringIO) -> pd.DataFrame:
    return pd.read_csv(
        csv,
        sep=";",
        index_col=0,
        decimal=",",
        header=0,
        skip_blank_lines=True,
        encoding="latin",
    )


class Process(Enum):
    """Enum for the process type."""

    SPRAYING = "Spraying"
    WIRE = "Wire"


def is_fielddata_match(
        found_fielddata: FieldData,
        expected_fielddata: dict[str, str],
) -> bool:
    for field_name, expected_value in expected_fielddata.items():
        found_field = next((o for o in found_fielddata if o.field_name == field_name), None)
        if found_field is None:
            return False
        
        found_value = found_field.value['name'] if found_field.field_type == "ComboBox" else found_field.value
        if found_value != expected_value:
            return False

    return True


class FURTHRmind:
    """FURTHRmind API interface."""

    __id: str
    __session: requests.Session
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
        self.__session = requests.Session()
        self.__session.headers.update({
            "X-API-KEY": config.furthr['api_key'],
            "Content-Type": "application/json"
        })
        self.fm = API(
            host=config.furthr['host'],
            api_key=config.furthr['api_key'],
            project_id=config.furthr['project_id']
        )

    @property
    def url(self) -> str:
        """Return the URL for the FURTHRmind API."""
        return config.furthr['host'] + "api2/{endpoint}"

    def get(self, endpoint: str) -> dict:
        """Get data from the FURTHRmind database."""
        url = self.url.format(endpoint=endpoint)
        return api_get(url, dict(self.__session.headers))

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
    
    def __download_item(
            self,
            override_selected: Experiment | Sample | ResearchItem | File | None,
            confirm_load: bool,
            download_fn: Callable[[str, dict], BytesIO | StringIO | requests.Response]
    ) -> tuple[BytesIO, str] | tuple[StringIO, str] | None:
        x = override_selected or self.__selected

        if x is None:
            return None
        
        if isinstance(x, Group):
            st.error("Cannot download group")
            return None
        
        if confirm_load and not button("Load", "load" + self.__id, stateful=True):
            return None
        
        if isinstance(x, File):
            b = download_fn(f"{config.furthr['host']}files/{x.id}", dict(self.__session.headers))
            if isinstance(b, (BytesIO, StringIO)):
                return (b, x.name)
            else:
                st.error(f"Failed to download file {x.name}: {b.reason}")
                return None
        
        # x must be a container when reaching this point
        if not x._fetched:
            x.get()

        files = x.files

        if self.file_extension:
            files = [f for f in files if f.name.endswith(self.file_extension)]

        if not files:
            if self.file_extension:
                st.error(f"No {self.file_extension} files found")
            else:
                st.error("No files found")
            return None
        
        downloaded_files = []
        bar_title = "Download in progress. Please wait."
        my_bar = st.progress(0, text=bar_title)

        for index, file in enumerate(files):
            downloaded_files.append(self.__download_item(file, False, download_fn))
            my_bar.progress(index / len(files), text=bar_title)

        my_bar.empty()
        return downloaded_files

    def download_bytes(
            self,
            override_selected: Experiment | Sample | ResearchItem | File | None = None,
            confirm_load: bool = True
    ) -> tuple[BytesIO, str] | None:
        """Download the selected item as bytes from the FURTHRmind database."""
        return self.__download_item(override_selected, confirm_load, api_get_bytes)
    
    def download_string(
            self,
            override_selected: Experiment | Sample | ResearchItem | File | None = None,
            confirm_load:bool = True
    ) -> tuple[BytesIO, str] | None:
        """Download the selected item as string from the FURTHRmind database."""
        return self.__download_item(override_selected, confirm_load, api_get_string)
    
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

    def upload_csv(self, csv: pd.DataFrame, name: str) -> None:
        """Upload a CSV file to the FURTHRmind database."""
        self.upload_file(lambda tmp_path: (
            csv.to_csv(tmp_path),
            name + ".csv"
        ))
