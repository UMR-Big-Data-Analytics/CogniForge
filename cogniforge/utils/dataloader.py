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
    header: int = 1
    skip_blank_lines: bool = True
    skiprows: int = 0
    columns: list = None
    with_preview: bool = True

    def _controls(self):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            self.delimiter = st.selectbox("Delimiter", [",", ";"], index=1)
        with col2:
            self.decimal = st.selectbox("Decimal", [",", "."], index=0)
        with col3:
            self.header = st.number_input("Data starts at row",
                                          value=1, min_value=1, step=1,
                                          help="Specify which row the data starts at")
        with col4:
            self.skiprows = st.number_input("Skip additional rows",
                                            min_value=0, value=0, step=1,
                                            help="Number of rows to skip after the starting row")

        self.skip_blank_lines = st.checkbox("Skip blank lines", value=True)

    def get_row_number(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add row number to df
        """
        df.index = range(1, len(df) + 1)
        df.index.name = 'No.'
        return df

    def _preview(self):
        """
        Preview method:
        - Read csv
        - Specify the first 2 rows as headers
        - Preview the first 10 rows of the df with specified data options
        """
        st.write("### Data Preview")
        self.csv.seek(0)
        total_rows = sum(1 for line in self.csv) - 2  # Subtract 2 for header /measurement units
        st.write(f"Total number of rows in dataset: {total_rows:,}")
        self.csv.seek(0)
        preview_df = self._read_csv(nrows=10)
        preview_df = self.get_row_number(preview_df)
        st.dataframe(preview_df, use_container_width=True, height=350)

    def _read_csv(self, nrows: int = None) -> pd.DataFrame:
        self.csv.seek(0)
        start_row = self.header + 1 # data starts after header
        rows_to_skip = list(range(2, start_row + self.skiprows + 1)) # calculate row to skip from starting row

        df = pd.read_csv(
            self.csv,
            sep=self.delimiter,
            decimal=self.decimal,
            header=[0, 1],
            skip_blank_lines=self.skip_blank_lines,
            skiprows=rows_to_skip if rows_to_skip else None,
            encoding="latin",
            nrows=nrows,
        )
        return self.get_row_number(df)

    def get_dataframe(self) -> pd.DataFrame:
        self._controls()
        if self.with_preview:
            self._preview()
        if button("Process Data", key="processdata", stateful=True):
            try:
                self.df = self._read_csv()
                return self.df
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.write(self.csv.read())