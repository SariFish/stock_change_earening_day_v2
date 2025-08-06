import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go

st.set_page_config(layout="wide")
st.title("Stock Earnings Explorer")

# Track session state for chart display and last symbol
if "show_charts" not in st.session_state:
    st.session_state["show_charts"] = False
if "last_symbol" not in st.session_state:
    st.session_state["last_symbol"] = ""

symbol = st.text_input("Enter a stock symbol (e.g., EQIX):", "EQIX").upper()

# Reset chart button if symbol changes
if st.session_state["last_symbol"] != symbol:
    st.session_state["show_charts"] = False
    st.session_state["last_symbol"] = symbol

if st.button("Show Charts"):
    st.session_state["show_charts"] = True

if st.session_state["show_charts"]:
    ticker = yf.Ticker(symbol)
    # Download 3y historical data
    hist = ticker.history(period='3y').reset_index()
    hist['Date'] = pd.to_datetime(hist['Date'])

    # Get earnings dates
    try:
        earnings_df = ticker.get_earnings_dates(limit=12)
        earnings_dates = pd.to_datetime(earnings_df.index)
    except Exception:
        st.error("Yahoo Finance is temporarily blocking data requests. Please wait a few minutes and try again.")
        st.stop()

    # === Chart 1: Percent Change After Earnings Reports ===
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
        y_pct, styles, hover_dates = [], [], []
        for label, days in offsets.items():
            found, min_days_diff, best_price = False, None, None
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

    # === Chart 2: Stock Price and Earnings Dates (by Month Color) ===

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
        dt_python = pd.to_datetime(dt).to_pydatetime()
        fig2.add_shape(
            type='line',
            x0=dt_python, x1=dt_python,
            y0=hist['Close'].min(), y1=hist['Close'].max(),
            line=dict(color=color, width=2, dash='dash')
        )
        fig2.add_annotation(
            x=dt_python,
            y=hist['Close'].max(),
            text=dt.strftime('%b-%Y'),
            showarrow=False,
            yshift=10,
            font=dict(color=color, size=9)
        )

    month_legend = [go.Scatter(
        x=[None], y=[None],
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

    # === Chart 3: Monthly High-Low Price Gap (%) by Month (Interactive) ===

    df = hist.copy()
    df['YearMonth'] = df['Date'].dt.to_period('M')
    df['MonthNum'] = df['Date'].dt.month
    df['WeekOfMonth'] = ((df['Date'].dt.day - 1) // 7) + 1

    monthly_summary = []
    for period, group in df.groupby('YearMonth'):
        try:
            high = group['High'].max()
            low = group['Low'].min()
            gap_pct = (high - low) / high * 100
            high_row = group[group['High'] == high].iloc[0]
            low_row = group[group['Low'] == low].iloc[0]
            # HighFirst: True if high comes before low in the month
            high_first = high_row['Date'] < low_row['Date']
            summary = {
                'YearMonth': str(period),
                'High': high,
                'High Date': high_row['Date'],
                'Low': low,
                'Low Date': low_row['Date'],
                'Gap %': gap_pct,
                'Low Week': low_row['WeekOfMonth'],
                'MonthNum': int(high_row['MonthNum']),
                'HighFirst': high_first
            }
            monthly_summary.append(summary)
        except Exception:
            pass

    monthly_summary_df = pd.DataFrame(monthly_summary)
    # Add the order info to the hover text
    monthly_summary_df['hovertext'] = monthly_summary_df.apply(
        lambda row: (
            f"Month: {row['YearMonth']}<br>"
            f"Gap: {row['Gap %']:.2f}%<br>"
            f"High: {row['High Date'].date()} (${row['High']:.2f})<br>"
            f"Low: {row['Low Date'].date()} (${row['Low']:.2f})<br>"
            f"Order: {'High before Low' if row['HighFirst'] else 'Low before High'}<br>"
            f"Low Week: {int(row['Low Week'])}"
        ),
        axis=1
    )

    st.subheader("3. Monthly High-Low Price Gap (%) by Month (Interactive)")
    months_map = {i: pd.Timestamp(month=i, day=1, year=2024).strftime('%B') for i in range(1, 13)}
    month_select = st.selectbox(
        "Select Month (all years):",
        options=sorted(monthly_summary_df['MonthNum'].unique()),
        format_func=lambda x: months_map[x]
    )

    # Define bar colors by HighFirst, highlight selected month as tomato
def get_bar_color(row, selected_month):
     if row['HighFirst']:
         return "#B4DAF5"  # light blue
     else:
         return "#FFD8B4"  # light orange

bar_colors = [get_bar_color(row, month_select) for idx, row in monthly_summary_df.iterrows()]
bar_opacity = [1.0 if row['MonthNum'] == month_select else 0.6 for idx, row in monthly_summary_df.iterrows()]

fig3 = go.Figure([
    go.Bar(
        x=monthly_summary_df['YearMonth'],
        y=monthly_summary_df['Gap %'],
        marker={
            'color': bar_colors,
            'opacity': bar_opacity
        },
        customdata=monthly_summary_df['MonthNum'],
        hovertext=monthly_summary_df['hovertext'],
        text=new_texts,
        textposition="auto",
        textfont=dict(size=20, color="black")
    )
])

fig3.update_traces(hovertemplate='%{hovertext}<extra></extra>')
    fig3.update_layout(
        title="Monthly High-Low Price Gap (%)",
        xaxis_title="Year-Month",
        yaxis_title="Gap (%)",
        width=1600,
        height=600,
        margin=dict(l=40, r=40, t=80, b=40)
    )

    st.plotly_chart(fig3, use_container_width=False)
