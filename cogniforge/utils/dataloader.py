from dataclasses import dataclass
from io import StringIO
import pandas as pd
import streamlit as st
import csv
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
    start_row: int = 0
    end_row: int = 1000

    def _controls(self):
        try:
            initial_df = pd.read_csv(
                self.csv,
                sep=self.delimiter,
                decimal=",",
                header=0,
                nrows=None,
                encoding="latin1"
            )
            total_rows = len(initial_df)
            st.write(f"Total rows in dataset: {total_rows:,}")

            use_full_dataset = st.checkbox("Use entire dataset", value=False,
                                           help="Check to use the complete dataset. Uncheck to specify a sample range.")

            if not use_full_dataset:
                col1, col2, col3 = st.columns(3)
                with col1:
                    self.header = st.number_input("Data starts at row", value=1, min_value=1, step=1,help="Specify the row where data begins")
                with col2:
                    self.skiprows = st.number_input("Skip additional data rows", min_value=0, value=0, step=1,help="Number of data rows to skip")
                with col3:
                    self.end_row = st.number_input("End Row", min_value=self.header + 1,value=min(1000, total_rows), max_value=total_rows, step=1,
                                                   help=f"Last row to include (maximum: {total_rows:,})")

                selected_rows = self.end_row - self.header
                st.write(f"Selected range: {selected_rows:,} rows (from row {self.header:,} to {self.end_row:,})")
            else:
                self.header = 1
                self.skiprows = 0
                self.end_row = total_rows
                st.write(f"Using complete dataset: {total_rows:,} rows")

            self.skip_blank_lines = st.checkbox("Skip blank lines", value=True)
            self.use_full_dataset = use_full_dataset

        except Exception as e:
            st.error(f"Error: {str(e)}")

    def _read_csv(self, nrows: int = None) -> pd.DataFrame:
        self.csv.seek(0)
        try:
            first_row = pd.read_csv(
                self.csv,
                sep=self.delimiter,
                nrows=1,
                encoding="latin"
            )
            self.csv.seek(0)
            # deal with header: combine variable name and the measurement unit
            if '[' in str(first_row.columns[0]):
                df = pd.read_csv(
                    self.csv,
                    sep=self.delimiter,
                    decimal=",",
                    header=0,
                    encoding="latin1",
                    thousands="."
                )
            else:
                df = pd.read_csv(
                    self.csv,
                    sep=self.delimiter,
                    decimal=",",
                    header=[0, 1],
                    encoding="latin1",
                    thousands="."
                )
                # Variable name and meausuremunit are on the same line!!!
                df.columns = [f"{col[0]}[{col[1]}]" if pd.notna(col[1]) else col[0] for col in df.columns]

            if nrows is not None:
                df = df.head(nrows)
            else:
                start_row = self.header
                rows_to_skip = list(range(1, start_row + self.skiprows + 1))
                # skippp
                if rows_to_skip:
                    df = df.iloc[len(rows_to_skip):]
                actual_end = min(self.end_row, len(df)) if self.end_row > 0 else len(df)
                df = df.iloc[self.start_row:actual_end]

            return df

        except Exception as e:
            st.error(f"CSV reading error: {str(e)}")
            return None

    # incase use a large dataset
    def _display_large_dataframe(self, df, page_size=1000):
        total_rows = len(df)
        pages = total_rows // page_size
        if total_rows % page_size != 0:
            pages += 1
        if pages > 1:
            page = st.number_input("Page", 1, pages, 1, help=f"Pick a page (total: {pages})")
        else:
            page = 1

        start = (page - 1) * page_size
        end = start + page_size
        if end > total_rows:
            end = total_rows
        st.write(f"Showing rows {start + 1} to {end} out of {total_rows}")

        page_df = df.iloc[start:end].copy()
        number_cols = page_df.select_dtypes(['float64', 'float32']).columns
        for col in number_cols:
            page_df[col] = page_df[col].apply(
                lambda x: f'{float(x):.6f}'.replace(',', '.') if pd.notna(x) else x
            )
        st.dataframe(page_df, use_container_width=True, height=350)

    def _preview(self):
        st.write("### Data Preview")
        self.csv.seek(0)
        total_rows = sum(1 for line in self.csv) - 2

        preview_text = (f"Previewing first 15 rows of the " + ("complete dataset" if self.use_full_dataset else "selected range") +
                        f" ({total_rows:,} total rows)")
        st.write(preview_text)

        self.csv.seek(0)
        preview_df = self._read_csv(nrows=15)

        if preview_df is not None:
            # Change decimal to dot
            display_df = preview_df.copy()
            numeric_cols = display_df.select_dtypes(include=['float64', 'float32']).columns
            for col in numeric_cols:
                display_df[col] = display_df[col].apply(
                    lambda x: f'{float(x):.6f}'.replace(',', '.') if pd.notnull(x) else x)

            st.dataframe(display_df, use_container_width=True, height=350)

    def get_dataframe(self) -> pd.DataFrame:
        try:
            self._controls()
            if self.with_preview:
                self._preview()

            if button("Process Data", key="processdata", stateful=True):
                try:
                    self.df = self._read_csv()
                    if self.df is not None and not self.df.empty:
                        st.write("### Processed Data")
                        if self.use_full_dataset:
                            dataset_info = f"Using complete dataset ({len(self.df):,} rows)"
                        else:
                            dataset_info = (f"Using selected range: {len(self.df):,} rows " f"(from row {self.start_row:,} to {self.end_row:,})")
                        st.write(dataset_info)

                        self._display_large_dataframe(self.df, page_size=1000)

                        csv_data = self.df.to_csv(
                            sep=self.delimiter,
                            decimal=self.decimal,
                            index=False
                        )
                        st.download_button(
                            label="Download processed data",
                            data=csv_data,
                            file_name="processed_data.csv",
                            mime="text/csv"
                        )

                        # Store for other ppages
                        st.session_state.processed_df = self.df.copy()
                        st.session_state.data_info = {
                            'total_rows': len(self.df),
                            'use_full_dataset': self.use_full_dataset,
                            'start_row': self.start_row,
                            'end_row': self.end_row,
                            'header': self.header,
                            'skiprows': self.skiprows
                        }
                        st.success("Data processed and ready for analysis!")
                        return self.df
                    else:
                        st.warning("No data to process")
                        return None
                except Exception as e:
                    st.error(f"Data processing error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Data loading error: {str(e)}")
            return None