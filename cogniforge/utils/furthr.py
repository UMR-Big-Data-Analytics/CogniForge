import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO, StringIO
import time

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from furthrmind import Furthrmind as API
from furthrmind.collection import Experiment, File, Group, Project, ResearchItem, Sample
from furthrmind.file_loader import FileLoader

from .object_select_box import selectbox
from .state_button import button

load_dotenv()


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


@dataclass
class FURTHRmind:
    """FURTHRmind API interface."""

    id: str = "furtherwidget"
    api_key: str | None = os.getenv("FURTHRMIND_API_KEY")
    host: str = "https://furthr.informatik.uni-marburg.de/"
    session: requests.Session = field(default_factory=requests.Session)
    process: Process = Process.SPRAYING
    file_type: str = "csv"

    def __post_init__(self):
        self.session.headers.update(
            {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        )

    @property
    def url(self) -> str:
        """Return the URL for the FURTHRmind API."""
        return self.host + "api2/{endpoint}"

    def get(self, endpoint: str) -> dict:
        """Get data from the FURTHRmind database."""
        url = self.url.format(endpoint=endpoint)
        return api_get(url, dict(self.session.headers))

    def download_string_file(self, file: File) -> StringIO | None:
        """Download a file from the FURTHRmind database."""
        placeholder = st.empty()
        placeholder.info("⏳ Fetching data... Please wait!")
        csv = api_get_string(f"{self.host}files/{file.id}", dict(self.session.headers))
        st.session_state.clicked["loaddownload"] = False
        placeholder.success("✅ Data loaded successfully!")
        time.sleep(1)
        placeholder.empty() 
        if isinstance(csv, StringIO):
            return csv
        st.error(f"Failed to download file {file.name}: {csv.reason}")

    def download_bytes_file(self, file: File) -> BytesIO | None:
        """Download a file from the FURTHRmind database."""
        b = api_get_bytes(f"{self.host}files/{file.id}", dict(self.session.headers))
        if isinstance(b, BytesIO):
            return b
        st.error(f"Failed to download file {file.name}: {b.reason}")

    def setup_project(self) -> API:
        """Setup the project for the API."""
        fm = API(
            host="https://furthr.informatik.uni-marburg.de/",
            api_key=os.getenv("FURTHRMIND_API_KEY"),
        )

        projects = Project.get_all()
        project = selectbox(
            projects, label="Choose a project", key=f"{self.id}_project"
        )

        fm = API(
            host="https://furthr.informatik.uni-marburg.de/",
            api_key=os.getenv("FURTHRMIND_API_KEY"),
            project_id=project.id,
        )
        return fm

    def select_group(self) -> Group | None:
        """Select a group from the project."""
        groups = Group.get_all()
        group = selectbox(groups, label="Choose a group", key=f"{self.id}_group")
        return group

    def select_experiment(
        self, group: Group
    ) -> Experiment | Sample | ResearchItem | None:
        """Select an experiment or sample from the group."""
        exp_sam = group.experiments + group.samples
        for l in list(group.researchitems.values()):
            exp_sam += l
        chosen_data: Experiment | Sample | None = selectbox(
            exp_sam,
            format_name=lambda o: o.name,
            label="Choose an experiment/sample",
            key=f"{self.id}_experiment",
        )
        if chosen_data is None:
            return None
        chosen_data = chosen_data.__class__.get(id=chosen_data.id)
        return chosen_data

    def select_file(self, chosen_data: Experiment | Sample) -> File | None:
        """Select a file from the experiment or sample."""
        files: list[File] = chosen_data.files
        files = [f for f in files if f.name.endswith(self.file_type)]
        file = selectbox(
            files, label=f"Choose a {self.file_type} file", key=f"{self.id}_file"
        )
        return file

    def download_bytes(self) -> tuple[BytesIO, str] | None:
        """Download any file from the FURTHRmind database."""
        _fm = self.setup_project()
        group = self.select_group()
        if group is not None:
            experiment = self.select_experiment(group)
            if experiment is not None:
                file = self.select_file(experiment)
                if file is not None:
                    if button("Load", "load" + self.id, stateful=True):
                        return (self.download_bytes_file(file), file.name)
                else:
                    st.error("No file found")
            else:
                st.error("No experiment or sample found")
        else:
            st.error("No group found")

    def download_csv(self) -> tuple[StringIO | None, str | None]:
        _fm = self.setup_project()
        group = self.select_group()
        if group is not None:
            experiment = self.select_experiment(group)
            if experiment is not None:
                file = self.select_file(experiment)
                if file is not None:
                    if button("Load", "load" + self.id, stateful=True):
                        df = self.download_string_file(file)
                        st.session_state.data = df
                        st.session_state.fileName = file.name
                        return df, file.name
        return None, None

    def upload_csv(self, csv: pd.DataFrame, name: str) -> None:
        """Upload a CSV file to the FURTHRmind database."""
        fm = self.setup_project()
        group = self.select_group()
        if group is not None:
            experiment = self.select_experiment(group)
            if experiment is not None:
                if button("Upload", "upload_anomaly_score",True):
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        csv_path = os.path.join(tmpdirname, f"{name}.csv")
                        csv.to_csv(csv_path)
                        file_loader = FileLoader(self.host, self.api_key)
                        file_loader.uploadFile(
                            csv_path,
                            parent={
                                "project": fm.project_id,
                                "type": "experiment",
                                "id": experiment.id,
                            },
                        )
                        st.success("Anomaly Score Uploaded Successfully!")
                        st.session_state.clicked["upload_anomaly_score"] = False
            else:
                st.error("No experiment or sample found")
        else:
            st.error("No group found")
