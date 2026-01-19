import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from enum import Enum

st.set_page_config(page_title="HSL Analysis", layout="wide")

# Available dates for selection
AVAILABLE_DATES = [
    "12/19/2025", "12/20/2025", "12/21/2025", "12/22/2025", "12/23/2025",
    "12/24/2025", "12/25/2025", "12/26/2025", "12/27/2025",
    "01/06/2026", "01/07/2026", "01/08/2026", "01/11/2026", "01/13/2026", "01/14/2026",
    "01/15/2026", "01/16/2026", "01/17/2026", "01/18/2026"
]

class PlotType(Enum):
    POWER_GENERATION = 0
    ERROR_RATES = 1


# Universal plot colors that work on both light and dark backgrounds
PLOT_COLORS = {
    "text_color": "#888888",  # Medium gray - visible on both
    "grid_color": "rgba(128,128,128,0.3)",  # Semi-transparent gray
    "annotation_bg": "rgba(128,128,128,0.5)",  # Semi-transparent gray
    "annotation_border": "rgba(128,128,128,0.6)",
}


def get_data_from_ignition(url: str):
    response = requests.get(url)
    return response.json()


def remove_curtailed_periods(df: pd.DataFrame):
    df = df[df["CURTAIL_FLAG"] == 'no waste']
    return df


def calculate_hourly_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate hourly averages for all error rate columns."""
    df = remove_curtailed_periods(df)
    error_columns = [
        "LATIMER_ERROR",
        "RUNE_1_MIN_SEQ_1MIN_STRIDE_ERROR",
    ]

    df["timestamp"] = pd.to_datetime(df["t_stamp"]).dt.tz_localize('UTC')
    df["timestamp"] = df["timestamp"].dt.tz_convert('America/Chicago')
    df["hour"] = df["timestamp"].dt.hour

    hourly_avg = df.groupby("hour")[error_columns].mean()

    return hourly_avg


def plot_actual_values(df: pd.DataFrame, date: str, plot_type: PlotType):
    """Plot actual values for ZIER HSL label vs. Rune HSL prediction using Plotly"""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["t_stamp"]).dt.tz_localize('UTC')
    df["timestamp"] = df["timestamp"].dt.tz_convert('America/Chicago')

    fig = go.Figure()

    # Highlight curtailment periods
    curtail_mask = df["CURTAIL_FLAG"] == "curtailment"

    num_curtailment_periods = 0
    num_rune_wins = 0

    if curtail_mask.any():
        curtail_starts = df.loc[curtail_mask & ~curtail_mask.shift(1, fill_value=False), "timestamp"]
        curtail_ends = df.loc[curtail_mask & ~curtail_mask.shift(-1, fill_value=False), "timestamp"]

        num_curtailment_periods = len(curtail_starts)

        # Add curtailment regions as shaded areas
        for i, (start, end) in enumerate(zip(curtail_starts, curtail_ends)):
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor="#B8860B", opacity=0.3,
                layer="below", line_width=0,
                name="Curtailment" if i == 0 else None,
                showlegend=(i == 0)
            )

        # Add markers where curtailment ends AND rune error < latimer error
        marker_timestamps = []
        marker_values = []
        for end in curtail_ends:
            post_curtail = df[df["timestamp"] > end]
            if not post_curtail.empty:
                first_post = post_curtail.iloc[0]
                rune_error = first_post["RUNE_1_MIN_SEQ_1MIN_STRIDE_ERROR"]
                latimer_error = first_post["LATIMER_ERROR"]
                if rune_error < latimer_error:
                    marker_timestamps.append(first_post["timestamp"])
                    if plot_type == PlotType.ERROR_RATES:
                        marker_values.append(rune_error)
                    else:
                        marker_values.append(first_post["RUNE_1_MIN_SEQ_1MIN_STRIDE"])

        num_rune_wins = len(marker_timestamps)

        if marker_timestamps:
            fig.add_trace(go.Scatter(
                x=marker_timestamps,
                y=marker_values,
                mode='markers',
                marker=dict(color='#00FF00', size=15, symbol='circle'),
                name='Rune < Latimer (post-curtail)'
            ))

    # Add main data traces
    if plot_type == PlotType.POWER_GENERATION:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["TENASKA_PV_GEN_NET"],
            mode='lines', name='Actual',
            line=dict(color='#FF6B35', width=1.5),
            opacity=0.8
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["LATIMER_MODEL"],
            mode='lines', name='Latimer (Predicted)',
            line=dict(color='#D4A000', width=1.5),
            opacity=0.7
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["RUNE_1_MIN_SEQ_1MIN_STRIDE"],
            mode='lines', name='Rune (Predicted)',
            line=dict(color='#00B4D8', width=1.5),
            opacity=0.8
        ))
        y_label = "Power Generation (MW)"
    else:  # ERROR_RATES
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["LATIMER_ERROR"],
            mode='lines', name='Latimer Error Rate',
            line=dict(color='#D4A000', width=1.5),
            opacity=0.7
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["RUNE_1_MIN_SEQ_1MIN_STRIDE_ERROR"],
            mode='lines', name='Rune Error Rate',
            line=dict(color='#00B4D8', width=1.5),
            opacity=0.8
        ))
        y_label = "Error Rate (%)"

    # Add statistics annotation
    if num_curtailment_periods > 0:
        pct = (num_rune_wins / num_curtailment_periods) * 100
        stats_text = (f"Curtailment Periods: {num_curtailment_periods}<br>"
                      f"Rune Wins: {num_rune_wins}<br>"
                      f"Win Rate: {pct:.1f}%")
        fig.add_annotation(
            x=0.02, y=0.98,
            xref='paper', yref='paper',
            text=stats_text,
            showarrow=False,
            font=dict(size=12, color=PLOT_COLORS["text_color"]),
            align='left',
            bgcolor=PLOT_COLORS["annotation_bg"],
            bordercolor=PLOT_COLORS["annotation_border"],
            borderwidth=1,
            borderpad=4
        )

    fig.update_layout(
        title=f"Actual vs Predicted Generation for {date}",
        xaxis_title="Time",
        yaxis_title=y_label,
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=PLOT_COLORS["text_color"]),
        xaxis=dict(gridcolor=PLOT_COLORS["grid_color"], zerolinecolor=PLOT_COLORS["grid_color"]),
        yaxis=dict(gridcolor=PLOT_COLORS["grid_color"], zerolinecolor=PLOT_COLORS["grid_color"]),
    )

    fig.update_xaxes(tickformat="%I:%M %p")

    return fig


def plot_hourly_averages(hourly_avg: pd.DataFrame, date: str):
    """Plot hourly averages for all error rate columns using Plotly."""
    fig = go.Figure()

    colors = {
        "LATIMER_ERROR": "#e63946",
        "RUNE_1_MIN_SEQ_1MIN_STRIDE_ERROR": "#0096c7",
    }

    labels = {
        "LATIMER_ERROR": "Latimer Error",
        "RUNE_1_MIN_SEQ_1MIN_STRIDE_ERROR": "Rune Error (1MIN_SEQ_1MIN_STRIDE)",
    }

    for column in hourly_avg.columns:
        fig.add_trace(go.Scatter(
            x=hourly_avg.index,
            y=hourly_avg[column],
            mode='lines+markers',
            name=labels.get(column, column),
            line=dict(color=colors.get(column)),
            marker=dict(size=8)
        ))

    fig.update_layout(
        title=f"Hourly Average Error Rates for {date}",
        xaxis_title="Hour",
        yaxis_title="Error Rate",
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=PLOT_COLORS["text_color"]),
        xaxis=dict(gridcolor=PLOT_COLORS["grid_color"], zerolinecolor=PLOT_COLORS["grid_color"]),
        yaxis=dict(gridcolor=PLOT_COLORS["grid_color"], zerolinecolor=PLOT_COLORS["grid_color"]),
    )

    fig.update_xaxes(
        tickmode='array',
        tickvals=hourly_avg.index,
        ticktext=[f"{h:02d}:00" for h in hourly_avg.index]
    )

    return fig


def collect_daily_error_rates(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Collect daily average error rates for the given date range."""
    dates = []
    latimer_errors = []
    rune_1min_seq_1min_stride_errors = []

    current_date = start_date
    while current_date <= end_date:
        formatted_date = current_date.strftime("%m/%d/%Y")

        try:
            data = get_data_from_ignition(f'https://admin.runehmi.com/system/webdev/dashboard/hsl-error?date={formatted_date}')
            df = pd.DataFrame(data["data"])

            dates.append(current_date)
            latimer_errors.append(df["LATIMER_ERROR"].mean())
            rune_1min_seq_1min_stride_errors.append(df["RUNE_1_MIN_SEQ_1MIN_STRIDE_ERROR"].mean())
        except Exception as e:
            st.warning(f"Error fetching data for {formatted_date}: {e}")

        current_date += timedelta(days=1)

    return pd.DataFrame({
        "date": dates,
        "LATIMER_ERROR": latimer_errors,
        "RUNE_1_MIN_SEQ_1MIN_STRIDE_ERROR": rune_1min_seq_1min_stride_errors,
    })


def plot_total_error_rates(df: pd.DataFrame):
    """Plot total error rates over the date range using Plotly."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["LATIMER_ERROR"],
        mode='lines+markers',
        name='Latimer',
        line=dict(color='#e63946', width=2),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["RUNE_1_MIN_SEQ_1MIN_STRIDE_ERROR"],
        mode='lines+markers',
        name='Rune (1MIN_SEQ_1MIN_STRIDE)',
        line=dict(color='#0096c7', width=2),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title="Daily Total Error Rates",
        xaxis_title="Date",
        yaxis_title="Average Error Rate",
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=PLOT_COLORS["text_color"]),
        xaxis=dict(gridcolor=PLOT_COLORS["grid_color"], zerolinecolor=PLOT_COLORS["grid_color"]),
        yaxis=dict(gridcolor=PLOT_COLORS["grid_color"], zerolinecolor=PLOT_COLORS["grid_color"]),
    )

    fig.update_xaxes(tickformat="%m/%d/%Y")

    return fig


# Main app
st.title("HSL Analysis Dashboard")

tab1, tab2 = st.tabs(["Single Date Analysis", "Total Analysis"])

with tab1:
    st.header("Single Date Analysis")

    selected_date = st.selectbox("Select Date", AVAILABLE_DATES, index=len(AVAILABLE_DATES) - 1)

    if st.button("Load Data", key="single_date_btn"):
        with st.spinner(f"Fetching data for {selected_date}..."):
            try:
                data = get_data_from_ignition(f'https://admin.runehmi.com/system/webdev/dashboard/hsl-error?date={selected_date}')
                df = pd.DataFrame(data["data"])

                # Display overall averages
                st.subheader("Overall Averages")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Latimer Error Rate", f"{df['LATIMER_ERROR'].mean():.3f}")
                with col2:
                    st.metric("Rune (1MIN_SEQ_1MIN_STRIDE) Error Rate", f"{df['RUNE_1_MIN_SEQ_1MIN_STRIDE_ERROR'].mean():.3f}")

                # Plot power generation
                st.subheader("Power Generation")
                fig_energy = plot_actual_values(df, selected_date, PlotType.POWER_GENERATION)
                st.plotly_chart(fig_energy, use_container_width=True)

                # Plot error rates
                st.subheader("Error Rates")
                fig_error = plot_actual_values(df, selected_date, PlotType.ERROR_RATES)
                st.plotly_chart(fig_error, use_container_width=True)

                # Plot hourly averages
                st.subheader("Hourly Average Error Rates")
                hourly_avg = calculate_hourly_averages(df)
                fig_hourly = plot_hourly_averages(hourly_avg, selected_date)
                st.plotly_chart(fig_hourly, use_container_width=True)

            except Exception as e:
                st.error(f"Error loading data: {e}")

with tab2:
    st.header("Total Analysis")
    st.write("Analyze error rates across all available dates")

    if st.button("Load All Data", key="total_btn"):
        with st.spinner("Fetching data for all dates..."):
            # Collect data for both date ranges
            df1 = collect_daily_error_rates(datetime(2025, 12, 19), datetime(2025, 12, 27))
            df2 = collect_daily_error_rates(datetime(2026, 1, 6), datetime(2026, 1, 8))
            daily_errors_df = pd.concat([df1, df2], ignore_index=True)

            # Display summary statistics
            st.subheader("Summary Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Latimer Error**")
                st.write(f"Mean: {daily_errors_df['LATIMER_ERROR'].mean():.3f}")
                st.write(f"Min: {daily_errors_df['LATIMER_ERROR'].min():.3f}")
                st.write(f"Max: {daily_errors_df['LATIMER_ERROR'].max():.3f}")
            with col2:
                st.write("**Rune Error (1MIN_SEQ_1MIN_STRIDE)**")
                st.write(f"Mean: {daily_errors_df['RUNE_1_MIN_SEQ_1MIN_STRIDE_ERROR'].mean():.3f}")
                st.write(f"Min: {daily_errors_df['RUNE_1_MIN_SEQ_1MIN_STRIDE_ERROR'].min():.3f}")
                st.write(f"Max: {daily_errors_df['RUNE_1_MIN_SEQ_1MIN_STRIDE_ERROR'].max():.3f}")

            # Plot daily error rates
            st.subheader("Daily Error Rates")
            fig_total = plot_total_error_rates(daily_errors_df)
            st.plotly_chart(fig_total, use_container_width=True)

            # Show raw data
            st.subheader("Raw Data")
            st.dataframe(daily_errors_df)
