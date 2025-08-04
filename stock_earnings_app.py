import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.title("Stock Earnings Explorer")

symbol = st.text_input("Enter a stock symbol (e.g., EQIX):", "EQIX").upper()

if st.button("Show Charts"):
    ticker = yf.Ticker(symbol)

    # Download historical price data (once)
    hist = ticker.history(period='3y').reset_index()
    hist['Date'] = pd.to_datetime(hist['Date'])

    # Download earnings dates (once)
    try:
        earnings_df = ticker.get_earnings_dates(limit=12)
        earnings_dates = pd.to_datetime(earnings_df.index)
    except Exception as e:
        st.error("Yahoo Finance is temporarily blocking data requests. Please wait a few minutes and try again.")
        st.stop()

    # --- First Chart: Percent Change After Earnings Reports ---
    offsets = {
        'Report Day': 0,
        'Mid 1st Week': 3,
        '1 Week After': 7,
        '1 Month After': 30,
        '3 Months After': 90,
        '6 Months After': 182,
        '1 Year After': 365
    }
    x_labels = list(offsets.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    fig = go.Figure()

    def get_style(days_diff):
        if days_diff <= 1:
            return 'solid'
        elif days_diff <= 3:
            return 'dash'
        else:
            return None

    for idx, report_date in enumerate(earnings_dates):
        base_row = hist[hist['Date'].dt.date == report_date.date()]
        if base_row.empty:
            base_row = hist[hist['Date'].dt.date == (report_date + pd.Timedelta(days=1)).date()]
            if base_row.empty:
                continue
        base_price = float(base_row['Close'].iloc[0])
        y_pct = []
        styles = []
        hover_dates = []

        for label, days in offsets.items():
            found = False
            min_days_diff = None
            best_price = None
            for diff in range(0, 4):
                for sign in [+1, -1]:
                    candidate_date = report_date + pd.Timedelta(days=days + sign * diff)
                    row = hist[hist['Date'].dt.date == candidate_date.date()]
                    if not row.empty:
                        price = float(row['Close'].iloc[0])
                        days_diff = abs((candidate_date.date() - (report_date + pd.Timedelta(days=days)).date()).days)
                        if min_days_diff is None or days_diff < min_days_diff:
                            found = True
                            min_days_diff = days_diff
                            best_price = price
                            best_diff = days_diff
                            best_candidate_date = candidate_date
                            best_actual_date = candidate_date.date()
                if found:
                    break
            if found:
                pct = (best_price - base_price) / base_price * 100
                y_pct.append(pct)
                styles.append(get_style(best_diff))
                hover_dates.append(str(best_actual_date))
            else:
                y_pct.append(None)
                styles.append(None)
                hover_dates.append("No data")

        month_year = report_date.strftime('%b %Y')
        fig.add_trace(go.Scatter(
            x=x_labels,
            y=y_pct,
            mode='lines+markers',
            name=f"{month_year}",
            line=dict(color=colors[idx % len(colors)], width=2, shape='linear', dash='solid'),
            marker=dict(size=8),
            customdata=[f"Trading Day: {d}" for d in hover_dates],
            hovertemplate=(
                "<b>Earnings Month:</b> %{meta}<br>"
                "<b>Interval:</b> %{x}<br>"
                "<b>Change:</b> %{y:.2f}%<br>"
                "%{customdata}<extra></extra>"
            ),
            meta=month_year,
            showlegend=True,
            connectgaps=False
        ))

    fig.add_hline(
        y=0,
        line_dash='dot',
        line_color='gray',
        annotation_text='0%',
        annotation_position='bottom right'
    )
    fig.add_annotation(
        x=0.055, y=0.95,
        xref="paper", yref="paper",
        align="left",
        showarrow=False,
        text=(
            "<b>Explanation:</b><br>"
            "<b>Solid line</b>: Trading day within ±1 day of target<br>"
            "<b>Dashed line</b>: Trading day within ±2-3 days<br>"
            "Points with no data are skipped<br>"
            "Hover for details"
        ),
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    fig.update_layout(
        title=f"{symbol}: Percent Change After Earnings Reports",
        xaxis_title="Time Interval",
        yaxis_title="Change (%) from Earnings Day",
        legend_title="Earnings Report Date",
        hovermode="closest",
        margin=dict(t=60, r=60)
    )
    st.subheader("1. Percent Change After Earnings Reports")
    st.plotly_chart(fig, use_container_width=True)

    # --- Second Chart: Stock Price and Earnings Dates (by Month Color) ---

    month_colors = {
        1: 'blue',     2: 'green',   3: 'orange',  4: 'purple',
        5: 'cyan',     6: 'brown',   7: 'pink',    8: 'olive',
        9: 'teal',    10: 'red',    11: 'navy',   12: 'magenta'
    }

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=hist['Date'],
        y=hist['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='royalblue')
    ))

    for dt in earnings_dates:
        month = dt.month
        color = month_colors.get(month, 'black')
        fig2.add_vline(
            x=str(dt.date()),  # <-- THE MAIN FIX HERE!
            line=dict(color=color, width=2, dash='dash'),
            annotation_text=dt.strftime('%b-%Y'),
            annotation_position='top left'
        )

    # Add legend entries for each month
    month_legend = [go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color=month_colors[m]),
        legendgroup=month_colors[m],
        showlegend=True,
        name=pd.Timestamp(month=m, day=1, year=2024).strftime('%b')
    ) for m in month_colors.keys()]

    for trace in month_legend:
        fig2.add_trace(trace)

    fig2.update_layout(
        title=f"{symbol} Stock Price — Last 3 Years (Earnings Dates Colored by Month)",
        xaxis_title="Date",
        yaxis_title="Close Price (USD)",
        legend_title="Legend",
        margin=dict(t=60, r=60)
    )

    st.subheader("2. Stock Price and Earnings Dates (by Month Color)")
    st.plotly_chart(fig2, use_container_width=True)
