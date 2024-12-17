import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go

def preprocess_data(filepath):
    """
    Preprocess data untuk forecasting.
    Mengembalikan data dan daftar kolom variabel.
    """
    # Baca data dari file CSV
    data = pd.read_csv(filepath)

    # Pastikan kolom waktu ada
    if "time" not in data.columns:
        raise ValueError("Data is missing required 'time' column")

    # Pastikan kolom waktu dalam format datetime
    data["time"] = pd.to_datetime(data["time"])
    data.set_index("time", inplace=True)

    # Isi nilai yang hilang (missing values) dengan rata-rata kolom
    data.fillna(data.mean(), inplace=True)

    # Ambil semua kolom selain "time"
    columns = data.columns.tolist()

    return data, columns

def train_and_forecast(filepath, variable, horizon):
    """
    Melakukan training dan forecasting pada data berdasarkan variabel dan horizon.
    Menghasilkan HTML plot untuk ditampilkan.
    """
    # Preprocessing data
    data, columns = preprocess_data(filepath)

    # Pilih variabel untuk forecasting
    if variable == "overall":
        variable = "total load actual"  # Mapping untuk variabel keseluruhan
    if variable not in columns:
        raise ValueError(f"Variable '{variable}' not found in data columns: {columns}")

    # Resampling data menjadi harian
    daily_data = data[variable].resample("D").sum()

    # Konfigurasi horizon
    if horizon == "6_months":
        steps = 180  # 6 bulan
    elif horizon == "1_year":
        steps = 365  # 1 tahun
    elif horizon == "5_years":
        steps = 1825  # 5 tahun
    else:
        raise ValueError("Invalid horizon. Choose '6_months', '1_year', or '5_years'.")

    # Holt-Winters Exponential Smoothing
    model = ExponentialSmoothing(daily_data, trend="add", seasonal="add", seasonal_periods=365)
    fitted_model = model.fit()

    # Forecasting
    forecast = fitted_model.forecast(steps=steps)
    forecast_dates = pd.date_range(daily_data.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")

    # Visualisasi
    fig = go.Figure()

    # Tambahkan data historis
    fig.add_trace(
        go.Scatter(
            x=daily_data.index,
            y=daily_data,
            mode="lines",
            name="Historical Data",
            line=dict(color="blue"),
        )
    )

    # Tambahkan data forecasting
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast,
            mode="lines",
            name="Forecast",
            line=dict(color="red", dash="dash"),
        )
    )

    # Layout plot
    fig.update_layout(
        title=f"Forecast for {variable} ({horizon.replace('_', ' ').capitalize()})",
        xaxis_title="Date",
        yaxis_title="Values",
        template="plotly",
        showlegend=True,
    )

    # Mengembalikan plot dalam bentuk HTML
    return fig.to_html(full_html=False)