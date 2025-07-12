import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import config
from utils.furthr import FURTHRmind
from utils.dataloader import DataLoader


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x).permute(1, 0, 2)

class TransformerEncoderRegressor(nn.Module):
    def __init__(self, num_features=4, d_model=512, window_size=202, num_heads=8, num_layers=2, dim_feedforward=512, dropout=0.1):
        super(TransformerEncoderRegressor, self).__init__()
        self.embedding = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=5000, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model * window_size, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)



def predict_and_append(model, df, window_size=202):
    if df.shape[1] < 4:
        raise ValueError("Data must have at least 4 columns.")
    data = df.iloc[:, :4].values  # Take first 202 rows and 4 columns
    num_samples = data.shape[0] // window_size
    predictions = []
    for i in range(num_samples):
        start = i * window_size
        end = start + window_size
        chunk = data[start:end]
        tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)  # shape: (1, 202, 4)
        with torch.no_grad():
            prediction = model(tensor).item()
        predictions.extend([prediction] * window_size)
    
    
    # For leftover rows at the end, if any
    remainder = data.shape[0] % window_size
    if remainder:
        predictions.extend([None] * remainder)  # Or extrapolate if needed

    df["Predicted Thickness"] = predictions
    return df

def load_model_from_bytes(model_bytes):
    model = TransformerEncoderRegressor()
    state_dict = torch.load(model_bytes, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model



st.title("Layer Thickness Tool")
st.write("Welcome! Upload your data from [FURTHRmind]({}) to analyze layer thickness.".format(config.furthr['host']))

# Session state
for key in ["fileName", "data", "files_loaded"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "files_loaded" else False

# File Downloader
with st.status("Download Data from FURTHRmind", expanded=True):
    downloader = FURTHRmind("download")
    downloader.file_extension = "csv"
    downloader.select_file()
    data, filename = downloader.download_string_button() or (None, None)

    if data:
        st.session_state.data = data
        st.session_state.fileName = filename
        st.success("Data downloaded!")

# Tabs for Data Preview and Prediction
if st.session_state.data:
    tabs = st.tabs(["Data Preview", "Layer Prediction"])
    with tabs[0]:
        dl = DataLoader(csv=st.session_state.data)
        df = dl.get_processedDataFrame()


    with tabs[1]:
        st.write("## Layer Thickness Prediction")

        model_downloader = FURTHRmind("model_download")
        model_downloader.file_extension = "pth"
        model_downloader.select_file()
        model_data, model_name = model_downloader.download_bytes_button() or (None, None)

        if model_data:
            try:
                model = load_model_from_bytes(model_data)
                st.success(f"Model '{model_name}' loaded successfully!")

                if st.button("Predict Layer Thickness"):
                    try:

                        result_df = predict_and_append(model, df)
                        st.write("### Prediction Results")
                        st.dataframe(result_df)

                        csv = result_df.to_csv(index=False).encode("utf-8")
                        st.download_button("Download with Predictions", csv, file_name="predicted_output.csv")

                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")

            except Exception as e:
                st.error(f"Prediction Error: {e}")
