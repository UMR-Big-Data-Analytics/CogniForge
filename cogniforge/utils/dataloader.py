from dataclasses import dataclass
from io import StringIO

import pandas as pd
import streamlit as st

from .state_button import button


@dataclass
class DataLoader:
    csv: StringIO
    df: pd.DataFrame = None
    delimiter: str = ";"
    decimal: str = ","
    header: int = 0
    skip_blank_lines: bool = True
    skiprows: list[int] = None
    columns: list = None
    with_preview: bool = True

    def _controls(self):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            self.delimiter = st.selectbox("Delimiter", [",", ";"], index=1)
        with col2:
            self.decimal = st.selectbox("Decimal", [",", "."], index=0)
        with col3:
            self.header = st.number_input("Header", value=0)
        with col4:
            self.skiprows = st.multiselect("Skip rows", options=[0, 1, 2], default=[1])

        self.skip_blank_lines = st.checkbox("Skip blank lines", value=True)

    def _preview(self):
        st.dataframe(self._read_csv(nrows=10))

    def _read_csv(self, nrows: int = None) -> pd.DataFrame:
        self.csv.seek(0)
        return pd.read_csv(
            self.csv,
            sep=self.delimiter,
            index_col=0,
            decimal=self.decimal,
            header=self.header,
            skip_blank_lines=self.skip_blank_lines,
            skiprows=self.skiprows,
            encoding="latin",
            nrows=nrows,
        )

    def get_dataframe(self) -> pd.DataFrame:
        self._controls()
        if self.with_preview:
            self._preview()
        if button("Load CSV", key="loadcsv", stateful=True):
            try:
                self.df = self._read_csv()
                return self.df

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.write(self.csv.read())
