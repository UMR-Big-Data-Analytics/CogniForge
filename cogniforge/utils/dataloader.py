from dataclasses import dataclass
from io import StringIO
import pandas as pd
import streamlit as st
from utils.state_button import button

#test test
@dataclass
class DataLoader:
    csv: StringIO
    df: pd.DataFrame = None
    delimiter: str = ";"
    decimal: str = ","
    header: int = 0
    skip_blank_lines: bool = True
    skiprows: int = 0
    columns: list = None
    with_preview: bool = True
    start_row: int = 0
    end_row: int = 1000

    def __post_init__(self):
        # Initialize a new state variable to track file changes
        if 'previous_file_timestamp' not in st.session_state:
            st.session_state.previous_file_timestamp = None
        if 'file_loaded' not in st.session_state:
            st.session_state.file_loaded = False
        if 'current_dataset_name' not in st.session_state:
            st.session_state.current_dataset_name = None
        if 'total_rows' not in st.session_state:
            st.session_state.total_rows = self._get_total_rows()
        if 'use_full_dataset' not in st.session_state:
            st.session_state.use_full_dataset = False
        if 'current_df' not in st.session_state:
            st.session_state.current_df = None
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'detrending_active' not in st.session_state:
            st.session_state.detrending_active = False
        if 'smoothing_active' not in st.session_state:
            st.session_state.smoothing_active = False
        if 'form_params' not in st.session_state:
            st.session_state.form_params = {
                'header': 1,
                'skiprows': 0,
                'end_row': min(1000, st.session_state.total_rows)
            }

    def reset_state(self):
        """Reset the state when switching dataset modes."""
        total_rows = st.session_state.total_rows
        use_full_dataset = st.session_state.use_full_dataset
        current_dataset_name = st.session_state.current_dataset_name
        st.session_state.update({
            'total_rows': total_rows,
            'use_full_dataset': use_full_dataset,
            'current_dataset_name': current_dataset_name,
            'current_df': None,
            'processing_complete': False,
            'preview_shown': False,
            'parameters_set': False,
            'detrending_active': False,
            'smoothing_active': False,
            'form_params': {
                'header': 1,
                'skiprows': 0,
                'end_row': min(1000, total_rows)
            }
        })


    def _get_total_rows(self) -> int:
        """Returns the total number of rows in the CSV file."""
        try:
            self.csv.seek(0)
            df = pd.read_csv(self.csv, sep=self.delimiter, encoding="latin1", header=None)
            return len(df)
        except Exception as e:
            st.error(f"Error counting rows: {str(e)}")
            return 0

    def _show_current_dataset_info(self):
        """Shows information about the currently loaded dataset and any active analyses"""
        warning_message = []
        if st.session_state.current_df is not None and st.session_state.processing_complete:
            if st.session_state.use_full_dataset:
                warning_message.append(f"• Using complete dataset ({len(st.session_state.current_df):,} rows)")
            else:
                start_row = self.header + self.skiprows
                warning_message.append(
                    f"• Using selected range from row {start_row:,} to {self.end_row:,} ({len(st.session_state.current_df):,} rows)")
            if 'detrending_active' in st.session_state and st.session_state.detrending_active:
                warning_message.append("• Active detrending analysis in progress")
            if 'smoothing_active' in st.session_state and st.session_state.smoothing_active:
                warning_message.append("• Active smoothing analysis in progress")
            return warning_message

    def _preview(self):
        """Shows a preview of the data"""
        st.write("### Data Preview")
        filename = getattr(self.csv, 'name', 'Unknown')
        st.write(f"Dataset: {filename}")
        st.write(f"Previewing first 10 rows of the dataset ({st.session_state.total_rows:,} total rows)")
        self.csv.seek(0)
        first_row = pd.read_csv(self.csv, sep=self.delimiter, nrows=1, encoding="latin1")
        self.csv.seek(0)
        header = 0 if '[' in str(first_row.columns[0]) else [0, 1]
        preview_df = pd.read_csv(
            self.csv,
            sep=self.delimiter,
            decimal=",",
            header=header,
            encoding="latin1",
            thousands=".",
            nrows=10
        )
        # Adjust columns
        if isinstance(header, list):
            preview_df.columns = [f"{col[0]}[{col[1]}]" if pd.notna(col[1]) else col[0] for col in preview_df.columns]
        preview_df = preview_df.map(
            lambda x: f'{float(x):.6f}'.replace(',', '.') if isinstance(x, (float, int)) and pd.notnull(x) else x)
        preview_df.index = range(1, len(preview_df) + 1)
        st.dataframe(preview_df, use_container_width=True, height=350)



    def _read_csv(self) -> pd.DataFrame:
        """Reads and processes the CSV file"""
        self.csv.seek(0)
        try:
            first_row = pd.read_csv(self.csv, sep=self.delimiter, nrows=1, encoding="latin1")
            self.csv.seek(0)
            has_combined_headers = '[' in str(first_row.columns[0])
            header = 0 if has_combined_headers else [0, 1]
            df = pd.read_csv(
                self.csv,
                sep=self.delimiter,
                decimal=",",
                header=header,
                encoding="latin1",
                low_memory=False,
                thousands=".",
                skip_blank_lines=self.skip_blank_lines
            )
            if not has_combined_headers:
                df.columns = [f"{col[0]}[{col[1]}]" if pd.notna(col[1]) else col[0] for col in df.columns]
            if not st.session_state.use_full_dataset:
                df = df.iloc[self.header + self.skiprows - 1: self.end_row - 1]
            return df
        except Exception as e:
            st.error(f"CSV reading error: {str(e)}")
            return None

    def _display_large_dataframe(self, df, page_size=1000):
        """Displays a large DataFrame"""
        if df is None:
            return

        total_rows = len(df)
        pages = total_rows // page_size
        if total_rows % page_size != 0:
            pages += 1

        if pages > 1:
            page = st.number_input("Page", 1, pages, 1, help=f"Pick a page (total: {pages})", key="page_input")
        else:
            page = 1

        start = (page - 1) * page_size
        end = start + page_size
        if end > total_rows:
            end = total_rows

        if not st.session_state.use_full_dataset:
            adjusted_end = end - (self.skiprows)
            adjusted_start = start + 1
            st.write(f"Showing rows {adjusted_start:,} to {adjusted_end:,} out of {total_rows - self.skiprows:,}")
        else:
            st.write(f"Showing rows {start + 1:,} to {end:,} out of {total_rows:,}")

        page_df = df.iloc[start:end].copy()
        page_df.index = range(1, len(page_df) + 1)

        number_cols = page_df.select_dtypes(['float64', 'float32']).columns
        for col in number_cols:
            page_df[col] = page_df[col].apply(
                lambda x: f'{float(x):.6f}'.replace(',', '.') if pd.notna(x) else x
            )

        st.dataframe(page_df, use_container_width=True, height=350)

    def _check_for_existing_analysis(self):
        """Check if there's existing analysis when changing datasets."""
        filename = getattr(self.csv, 'name', 'Unknown Dataset')
        if (st.session_state.current_dataset_name is None or
                st.session_state.current_dataset_name == filename):
            return True
        if (st.session_state.get('current_df') is not None and
                (st.session_state.get('detrending_active') or
                 st.session_state.get('smoothing_active'))):
            st.warning("⚠️ Loading new data will clear all existing analyses.")
            proceed = st.button("Proceed with new data")
            return proceed
        return True


    def _process_submitted_data(self, parameters_changed):
        """Process the submitted data"""
        try:
            if parameters_changed:
                self.reset_state()
            with st.spinner("Processing data..."):
                filename = getattr(self.csv, 'name', 'Unknown Dataset')
                if st.session_state.current_dataset_name != filename:
                    st.info(f"Dataset changed: {filename}")
                    st.session_state.total_rows = self._get_total_rows()
                    self.reset_state()

                st.session_state.current_dataset_name = filename
                self.df = self._read_csv()

                if self.df is not None and not self.df.empty:
                    st.session_state.current_df = self.df.copy()
                    st.session_state.processing_complete = True

                    st.write("### Processed Data")
                    dataset_info = (
                        f"Using complete dataset ({len(self.df):,} rows)" if st.session_state.use_full_dataset
                        else f"Using selected range: {len(self.df):,} rows (from row {self.header + self.skiprows:,} to {self.end_row:,})")
                    st.write(dataset_info)

                    self._display_large_dataframe(self.df, page_size=1000)
                    st.success("Data processed and ready for analysis!")
                    return True
                else:
                    st.warning("No data was processed. Please check your parameters and try again.")
                    return False

        except Exception as e:
            st.error(f"Data processing error: {str(e)}")
            return False

    def get_dataframe(self) -> pd.DataFrame:
        try:
            filename = getattr(self.csv, 'name', 'Unknown Dataset')

            if st.session_state.current_dataset_name != filename:
                st.info(f"New dataset detected: {filename}")
                if not self._check_for_existing_analysis():
                    return None

                st.session_state.total_rows = self._get_total_rows()
                self.reset_state()
                st.session_state.current_dataset_name = filename

            if not st.session_state.get('parameters_set', False):
                self._preview()

            st.write("### Dataset Selection")
            st.write(f"Dataset: {filename}")
            total_rows = st.session_state.total_rows
            st.write(f"Total rows in dataset: {total_rows:,}")

            use_full_dataset = st.checkbox(
                "Use entire dataset",
                value=st.session_state.use_full_dataset,
                help="Check to use the complete dataset. Uncheck to specify a sample range.",
                key="dataset_mode"
            )

            if use_full_dataset != st.session_state.use_full_dataset:
                st.session_state.use_full_dataset = use_full_dataset
                st.session_state.form_params = {
                    'header': 1,
                    'skiprows': 0,
                    'end_row': total_rows if use_full_dataset else min(1000, total_rows)
                }

            with st.form("dataset_selection_form", clear_on_submit=False):
                if not use_full_dataset:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        header = st.number_input(
                            "Data starts at row",
                            value=st.session_state.form_params['header'],
                            min_value=1,
                            step=1,
                            help="Specify the row where data begins",
                            key="header_input"
                        )
                    with col2:
                        skiprows = st.number_input(
                            "Skip additional data rows",
                            min_value=0,
                            value=st.session_state.form_params['skiprows'],
                            step=1,
                            help="Number of data rows to skip",
                            key="skiprows_input"
                        )
                    with col3:
                        min_end_row = header + skiprows + 1
                        end_row = st.number_input(
                            "End Row",
                            min_value=min_end_row,
                            value=max(min_end_row, st.session_state.form_params['end_row']),
                            max_value=total_rows,
                            step=1,
                            help=f"Last row to include (maximum: {total_rows:,})",
                            key="end_row_input"
                        )

                    actual_start_row = header + skiprows
                    actual_row_count = end_row - actual_start_row
                    st.write(
                        f"Selected range: {actual_row_count:,} rows (from row {actual_start_row:,} to {end_row:,})")
                else:
                    header = 1
                    skiprows = 0
                    end_row = total_rows
                    st.write(f"Using complete dataset: {total_rows:,} rows")

                skip_blank_lines = st.checkbox("Skip blank lines", value=True)
                submitted = st.form_submit_button("Process Data")

            if submitted:
                self.header = header
                self.skiprows = skiprows
                self.end_row = end_row
                self.skip_blank_lines = skip_blank_lines

                st.session_state.form_params = {
                    'header': header,
                    'skiprows': skiprows,
                    'end_row': end_row
                }

                if self._process_submitted_data(True):
                    st.session_state.parameters_set = True
                    return self.df
            elif st.session_state.processing_complete and st.session_state.current_df is not None:
                st.write("### Current Processed Data")
                self._display_large_dataframe(st.session_state.current_df, page_size=1000)

            return None

        except Exception as e:
            st.error(f"Data loading error: {str(e)}")
            return None
    
    def get_processedDataFrame(self) -> pd.DataFrame:
        try:
            st.info(f"New dataset detected: {st.session_state.fileName}")

            # Ensure total rows are updated
            st.session_state.total_rows = self._get_total_rows()
            self.reset_state()
            if "use_full_dataset" not in st.session_state:
                st.session_state.use_full_dataset = False
            # Checkbox to show entire dataset or only the first 10 rows
            use_full_dataset = st.checkbox(
                "Show entire dataset",
                value=st.session_state.use_full_dataset,
                help="Check to see the complete dataset. Uncheck to preview only first 10 rows.",
                key="dataset_mode"
            )

            # If user toggles the checkbox, update session state
            if use_full_dataset != st.session_state.use_full_dataset:
                st.session_state.use_full_dataset = use_full_dataset
                st.rerun() 

            # Load the dataset
            self.csv.seek(0)
            df = pd.read_csv(
                self.csv,
                sep=self.delimiter,
                decimal=",",
                header=0,
                encoding="latin1",
                low_memory=False,
                skip_blank_lines=True,
                index_col=0
            )
            previewDf = df
            if previewDf.index.name is not None:
                previewDf.reset_index(inplace=True)
            # If full dataset is not selected, show only first 10 rows
            if not use_full_dataset:
                previewDf = df.head(10)

            # Store the dataframe in session state
            st.session_state.current_df = df.copy()
            dataset_info = (
                f"Showing complete dataset ({len(df):,} rows)"
                if use_full_dataset
                else f"Previewing first 10 rows of the dataset"
            )
            st.write(dataset_info)

            # Paginate and display the dataset (only if full dataset is selected)
            if use_full_dataset:
                self._display_large_dataframe(previewDf, page_size=1000)
            else:
                st.dataframe(previewDf, use_container_width=True, height=350)
            return df

        except Exception as e:
            st.error(f"Data loading error: {str(e)}")
            return None
