import os
import tempfile
from enum import Enum
from io import BytesIO, StringIO
import itertools

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

    id: str
    session: requests.Session
    process: Process = Process.SPRAYING
    file_type: str
    fm: API

    def __init__(self, id: str = "furtherwidget", file_type: str = "csv"):
        """Setup the project for the API."""
        self.id = id
        self.file_type = file_type
        self.session = requests.Session()
        self.session.headers.update(
            {"X-API-KEY": config.furthr['api_key'], "Content-Type": "application/json"}
        )
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
        return api_get(url, dict(self.session.headers))

    def select_group(self) -> Group | None:
        """Select a group from the project."""
        groups = self.fm.Group.get_all()
        group = selectbox(groups, label="Choose a group", key=f"{self.id}_group")
        if group is None:
            st.error("No group found")
        return group
    
    def select_item(
            self,
            group: Group | None,
            category: str | None = None, # only for research items
            expected_fielddata: dict[str, str] | None = None
    ) -> Experiment | Sample | ResearchItem | None:
        """Select a research item, experiment or sample from the group which has a specific attribute and field data"""
        if group is None:
            group = self.select_group()
            if group is None:
                return None

        if category:
            items = group.researchitems.get(category, [])
            label = f"Choose a {category} item"
        else:
            items = group.experiments + group.samples
            items.extend(itertools.chain.from_iterable(group.researchitems.values()))
            label = "Choose an experiment/sample/research item"

        if expected_fielddata:
            # fielddata not populated without get
            items = [
                o for o in items
                if is_fielddata_match(o.get().fielddata, expected_fielddata)
            ]
        
        return selectbox(
            items, label=label, key=f"{self.id}_item"
        )

    def select_file(self, chosen_data: Experiment | Sample | ResearchItem | None) -> File | None:
        """Select a file from the experiment or sample."""
        if chosen_data is None:
            chosen_data = self.select_item(None)
            if chosen_data is None:
                return None

        files: list[File] = chosen_data.files
        files = [f for f in files if f.name.endswith(self.file_type)]
        file = selectbox(
            files, label=f"Choose a {self.file_type} file", key=f"{self.id}_file"
        )
        if file is None:
            st.error("No file found")
        return file

    def download_bytes(self, file: File | None, confirm_load=True) -> tuple[BytesIO, str] | None:
        """Download any file from the FURTHRmind database."""
        if file is None:
            file = self.select_file(None)

        if file is not None and (not confirm_load or button("Load", "load" + self.id, stateful=True)):
            b = api_get_bytes(f"{config.furthr['host']}files/{file.id}", dict(self.session.headers))
            if isinstance(b, BytesIO):
                return (b, file.name)
            st.error(f"Failed to download file {file.name}: {b.reason}")

    def download_item(self, chosen_data: Experiment | Sample | ResearchItem | None) -> list[tuple[BytesIO, str]] | None:
        """Download experiment folder from the FURTHRmind database."""
        if chosen_data is None:
            chosen_data = self.select_item(None)
            if chosen_data is None:
                return None
        
        if not chosen_data._fetched:
            chosen_data.get()

        files = [f for f in chosen_data.files if f.name.endswith(self.file_type)]
        if not files:
            st.error("No files found")
            return None
        
        if button("Load", "load" + self.id, stateful=True):
            res = []
            bar_title = "Download in progress. Please wait."
            my_bar = st.progress(0, text=bar_title)

            for index, file in enumerate(files):
                res.append(self.download_bytes(file, confirm_load=False))
                my_bar.progress(index / len(files), text=bar_title)

            my_bar.empty()
            return res

    def download_csv(self, file: File | None, confirm_load=True) -> StringIO | None:
        """Download a CSV file from the FURTHRmind database."""
        if file is None:
            file = self.select_file(None)

        if file is not None and (not confirm_load or button("Load", "load" + self.id, stateful=True)):
            csv = api_get_string(f"{config.furthr['host']}files/{file.id}", dict(self.session.headers))
            if isinstance(csv, StringIO):
                return csv
            st.error(f"Failed to download file {file.name}: {csv.reason}")

    def upload_csv(self, csv: pd.DataFrame, name: str) -> None:
        """Upload a CSV file to the FURTHRmind database."""
        experiment = self.select_experiment(None)
        if experiment is not None and st.button("Upload", "upload_anomaly_score"):
            with tempfile.TemporaryDirectory() as tmpdirname:
                csv_path = os.path.join(tmpdirname, f"{name}.csv")
                csv.to_csv(csv_path)
                file_loader = FileLoader(config.furthr['host'], config.furthr['api_key'])
                file_loader.uploadFile(
                    csv_path,
                    parent={
                        "project": config.furthr['project_id'],
                        "type": "experiment",
                        "id": experiment.id,
                    },
                )
