import streamlit as st
import pandas as pd
import yfinance as yf
import time
import requests
from datetime import datetime
import pytz
import numpy as np
import plotly.graph_objs as go
from calendar import monthrange
import openai
import re

# --- API Keys from Streamlit Secrets ---
openai_api_key = st.secrets["OPENAI_API_KEY"]
polygon_api_key = st.secrets["POLYGON_API_KEY"]

# --- Helper: Fetch minute data from Polygon API ---
def fetch_polygon_minute_data(symbol, year, month, api_key):
    start_date = f"{year}-{str(month).zfill(2)}-01"
    end_date = f"{year}-{str(month).zfill(2)}-{monthrange(year, month)[1]}"
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key
    }
    resp = requests.get(url, params=params)
    data = resp.json()
    if "results" not in data or not data["results"]:
        return None
    eastern = pytz.timezone("America/New_York")
    df = pd.DataFrame(data["results"])
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.tz_convert(eastern)
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    return df

# --- Helper: Determine session type (pre, regular, after) ---
def get_session(row):
    h, m = row['hour'], row['minute']
    if h < 4 or h > 20 or (h == 20 and m > 0):
        return "none"
    elif h < 9 or (h == 9 and m < 30):
        return "pre"
    elif (h == 9 and m >= 30) or (9 < h < 16) or (h == 16 and m == 0):
        return "regular"
    elif (h > 16) or (h == 16 and m > 0) or (h < 20):
        return "after"
    elif h == 20 and m == 0:
        return "after"
    else:
        return "none"

# --- Helper: Get tickers for selected index ---
def get_tickers_by_index(index_name):
    if index_name == "NASDAQ-100":
        url = "https://en.wikipedia.org/wiki/NASDAQ-100"
        table = pd.read_html(url)[4]
        tickers = sorted(table['Ticker'].unique())
    elif index_name == "S&P 500":
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table = pd.read_html(url)[0]
        tickers = sorted(table['Symbol'].unique())
    else:
        tickers = []
    return tickers

# --- Remove all Markdown italics from AI output ---
def remove_all_markdown(text):
    # Remove bold (**text** or __text__)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    # Remove italics (*text* or _text_) but NOT already bold (because those were already stripped above)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    # Remove any leftover unmatched _text_ or *text* (greedy, for buggy outputs)
    text = re.sub(r'(_|\*)([^\*_]+)\1', r'\2', text)
    # Remove inline code formatting `like this`
    text = re.sub(r'`([^`]*)`', r'\1', text)
    return text

# --- Streamlit UI configuration ---
st.set_page_config(layout="wide")
st.title("AI Stock Recommendation App")

index_options = ["NASDAQ-100", "S&P 500"]
index_name = st.sidebar.selectbox("Select stock index (universe):", index_options)

with st.spinner(f"Loading tickers for {index_name}..."):
    tickers = get_tickers_by_index(index_name)

min_price = st.sidebar.number_input("Minimum current price ($):", min_value=0.0, value=0.0)
max_price = st.sidebar.number_input("Maximum current price ($): (0 = No limit)", min_value=0.0, value=0.0)
min_growth = st.sidebar.number_input("Minimum 3-month growth (%):", min_value=-100.0, value=0.0)
max_rec = st.sidebar.number_input(
    "Maximum average recommendation score (0 = No limit):",
    min_value=0.0, max_value=5.0, value=0.0,
    help="Lower score = stronger buy consensus. (1=Strong Buy, 3=Hold, 5=Strong Sell)"
)
top_n = st.sidebar.number_input("Max results to show:", min_value=1, max_value=100, value=20)

if "screening_results" not in st.session_state:
    st.session_state.screening_results = None

if st.button("Run Screening"):
    results = []
    st.info("This may take a few minutes (especially for S&P 500)...")
    progress = st.progress(0)
    for i, ticker in enumerate(tickers):
        try:
            stock = yf.Ticker(ticker)
            earnings = stock.get_earnings_dates(limit=6)
            if earnings.empty:
                continue

            today = pd.Timestamp.now(tz='UTC').normalize()
            three_months_ago = today - pd.Timedelta(days=90)
            report_date = None
            for d in earnings.index[::-1]:
                if abs((pd.to_datetime(d).normalize() - three_months_ago).days) <= 21:
                    report_date = pd.to_datetime(d).date()
                    break
            if not report_date:
                continue

            hist = stock.history(period="6mo")
            if hist.empty:
                continue
            hist = hist.reset_index()

            close_at_report = hist[hist['Date'].dt.date == report_date]
            if close_at_report.empty:
                try_next = hist[hist['Date'].dt.date > report_date].head(1)
                if try_next.empty:
                    continue
                close_at_report = try_next
            report_price = float(close_at_report['Close'].iloc[0])
            last_price = float(hist['Close'].iloc[-1])
            pct = 100 * (last_price - report_price) / report_price

            if min_price > 0 and last_price < min_price:
                continue
            if max_price > 0 and last_price > max_price:
                continue
            if pct < min_growth:
                continue

            info = stock.info
            low_target = info.get("targetLowPrice")
            median_target = info.get("targetMedianPrice")
            mean_target = info.get("targetMeanPrice")
            high_target = info.get("targetHighPrice")
            n_analysts = info.get("numberOfAnalystOpinions", 0)
            rec_mean = info.get("recommendationMean", None)
            analyst_url = f"https://finance.yahoo.com/quote/{ticker}/analysis"

            if max_rec > 0 and (rec_mean is None or rec_mean > max_rec):
                continue

            results.append({
                "Ticker": ticker,
                "Report Date": str(report_date),
                "Report Price": f"${report_price:,.2f}",
                "Current Price": f"${last_price:,.2f}",
                "Change (%)": f"{pct:.2f}%",
                "Min Target Price": f"${low_target:,.2f}" if low_target else "—",
                "Median Target Price": f"${median_target:,.2f}" if median_target else "—",
                "Max Target Price": f"${high_target:,.2f}" if high_target else "—",
                "Recommendation": f"{rec_mean:.2f} ({n_analysts})" if rec_mean else "—",
                "Analyst Link": f"[Analyst Page]({analyst_url})"
            })
        except Exception as e:
            continue
        progress.progress((i + 1) / len(tickers))
        time.sleep(0.2)
    progress.empty()
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("Change (%)", ascending=False).head(top_n)
        st.session_state.screening_results = df
    else:
        st.session_state.screening_results = None

df = st.session_state.screening_results
if df is not None:
    st.success(f"Found {len(df)} stocks matching your filters. Clickable analyst links in table. Select up to 3 for timing analysis below.")

    # Show screening table with clickable links (Markdown)
    st.write(df.to_markdown(index=False), unsafe_allow_html=False)
    st.caption("""
    **Recommendation Scale:** 1.0 = Strong Buy, 2.0 = Buy, 3.0 = Hold, 4.0 = Underperform, 5.0 = Strong Sell  
    (Number in parentheses = number of analysts in the average)
    """)

    # --- AI Recommendation (password-protected, using OpenAI 1.x syntax) ---
    if "ai_password_verified" not in st.session_state:
        st.session_state["ai_password_verified"] = False

    with st.expander("🔒 AI Recommendation (private access)"):
        password = st.text_input("Enter access code to use AI recommendation:", type="password")
        if st.button("Verify Code"):
            if password == st.secrets["AI_FEATURE_CODE"]:
                st.session_state["ai_password_verified"] = True
                st.success("Access granted. You can now run the AI analysis!")
            else:
                st.session_state["ai_password_verified"] = False
                st.error("Wrong code. Try again.")

        if st.session_state["ai_password_verified"]:
            if st.button("Ask AI for Table Recommendation"):
                cols = [
                    "Ticker", "Change (%)", "Current Price", "Recommendation",
                    "Min Target Price", "Median Target Price", "Max Target Price"
                ]
                summary_df = df[cols].to_string(index=False)
                prompt = (
                    "You are an expert stock analyst. Analyze the following stock screening table, "
                    "and recommend the top 1-3 stocks to consider buying, with a short reason for each choice. "
                    "Focus on percent growth, recommendation score, and price targets. "
                    "Be concise and clear for an investor.\n\n"
                    f"{summary_df}\n\n"
                    "List your recommendations as:\n1. ...\n2. ...\n3. ... (if relevant).\n also have break it into sub bullets as well."
                    "Please write your answer in plain text, do not use Markdown formatting such as *, _, or **."
                    "Please write your answer in plain text and do not use *any* formatting such as italics, bold, Markdown, LaTeX, or HTML."
                )
                with st.spinner("AI is analyzing your table..."):
                    client = openai.OpenAI(api_key=openai_api_key)
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful investment analyst."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=350,
                        temperature=0.4,
                    )
                    ai_response = response.choices[0].message.content

                def strip_all_formatting(text):
                    text = re.sub(r'([0-9])[_*]+', r'\1 ', text)
                    text = re.sub(r'(\*{1,2}|_{1,2})([^\*_]+?)\1', r'\2', text)
                    text = re.sub(r'\\textit\{([^}]*)\}', r'\1', text)
                    text = re.sub(r'\\emph\{([^}]*)\}', r'\1', text)
                    text = re.sub(r'[_*]+([A-Za-z0-9 ,.\-%$]+)[_*]+', r'\1', text)
                    text = re.sub(r'<[^>]+>', '', text)
                    text = re.sub(r'\$+([^$]+)\$+', r'\1', text)
                    text = re.sub(r' +', ' ', text)
                    return text.strip()

                cleaned_response = strip_all_formatting(ai_response)
                st.info(cleaned_response)

    # === Earnings Charts Section (NEW) ===
    st.header("Earnings Analysis & Charts")
    chart_stocks = st.multiselect(
        "Select up to 3 stocks for Earnings & Price Charts:",
        options=df["Ticker"].tolist(),
        max_selections=3,
        key="earnings_charts_select"
    )

    for symbol in chart_stocks:
        st.markdown(f"### {symbol} Earnings & Price Charts")
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='3y').reset_index()
            hist['Date'] = pd.to_datetime(hist['Date'])
            earnings_df = ticker.get_earnings_dates(limit=12)
            earnings_dates = pd.to_datetime(earnings_df.index)
        except Exception:
            st.error(f"Yahoo Finance is temporarily blocking data requests for {symbol}. Try again later.")
            continue

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

        df_hist = hist.copy()
        df_hist['YearMonth'] = df_hist['Date'].dt.to_period('M')
        df_hist['MonthNum'] = df_hist['Date'].dt.month
        df_hist['WeekOfMonth'] = ((df_hist['Date'].dt.day - 1) // 7) + 1

        monthly_summary = []
        for period, group in df_hist.groupby('YearMonth'):
            try:
                high = group['High'].max()
                low = group['Low'].min()
                gap_pct = (high - low) / high * 100
                high_row = group[group['High'] == high].iloc[0]
                low_row = group[group['Low'] == low].iloc[0]
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
        months_map = {i: pd.Timestamp(month=i, day=1, year=2024).strftime('%B') for i in range(1, 13)}
        st.subheader("3. Monthly High-Low Price Gap (%) by Month (Interactive)")
        month_select = st.selectbox(
            f"Select Month for {symbol} (all years):",
            options=sorted(monthly_summary_df['MonthNum'].unique()),
            format_func=lambda x: months_map[x],
            key=f"{symbol}_gap_month"
        )

        def get_bar_color(row, selected_month):
            if row['MonthNum'] == month_select:
                return "tomato"
            elif row['HighFirst']:
                return "#B4DAF5"  # light blue: High before Low
            else:
                return "#FFD8B4"  # light orange: Low before High

        new_colors = [get_bar_color(row, month_select) for idx, row in monthly_summary_df.iterrows()]
        new_texts = [
            f"{gap:.2f}%" if mn == month_select else ""
            for gap, mn in zip(monthly_summary_df['Gap %'], monthly_summary_df['MonthNum'])
        ]

        fig3 = go.Figure([
            go.Bar(
                x=monthly_summary_df['YearMonth'],
                y=monthly_summary_df['Gap %'],
                marker={'color': new_colors},
                customdata=monthly_summary_df['MonthNum'],
                hovertext=monthly_summary_df.apply(
                    lambda row: (
                        f"Month: {row['YearMonth']}<br>"
                        f"Gap: {row['Gap %']:.2f}%<br>"
                        f"High: {row['High Date'].date()} (${row['High']:.2f})<br>"
                        f"Low: {row['Low Date'].date()} (${row['Low']:.2f})<br>"
                        f"Order: {'High before Low' if row['HighFirst'] else 'Low before High'}<br>"
                        f"Low Week: {int(row['Low Week'])}"
                    ),
                    axis=1
                ),
                text=new_texts,
                textposition="auto",
                textfont=dict(
                    size=20,    
                    color="black"
                )
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

    # --- Multi-select for Polygon analysis ---
    available_tickers = df["Ticker"].tolist()
    if "selected_stocks" not in st.session_state or st.session_state.screening_results is None:
        st.session_state.selected_stocks = []
    selected = st.multiselect(
        "Select up to 3 stocks for timing analysis (Polygon API):",
        options=available_tickers,
        default=st.session_state.selected_stocks,
        max_selections=3
    )
    st.session_state.selected_stocks = selected

    if len(selected) > 0:
        st.header("Polygon Timing Analysis")
        today = datetime.now()
        for ticker in selected:
            st.subheader(f"{ticker} Timing Analysis")

            default_month = today.month - 1 if today.month > 1 else 12
            default_year = today.year if today.month > 1 else today.year - 1

            year = st.number_input(
                f"Year for {ticker}:", min_value=2017, max_value=today.year,
                value=default_year, key=f"{ticker}_year"
            )
            month = st.selectbox(
                f"Month for {ticker}:", list(range(1, 13)),
                index=default_month - 1,
                format_func=lambda m: datetime(2000, m, 1).strftime('%B'),
                key=f"{ticker}_month"
            )
            run_polygon = st.button(f"Run Polygon Price Analysis for {ticker}", key=f"{ticker}_run")

            if run_polygon:
                with st.spinner(f"Fetching price data for {ticker}..."):
                    df_polygon = fetch_polygon_minute_data(ticker, year, month, polygon_api_key)
                    if df_polygon is None or df_polygon.empty:
                        st.warning("No data available from Polygon for this period.")
                        continue

                    df_polygon['session'] = df_polygon.apply(get_session, axis=1)
                    results = []
                    for date, group in df_polygon.groupby('date'):
                        result = {'Date': date}
                        regular = group[group['session'] == 'regular']
                        if not regular.empty:
                            regular_sorted = regular.sort_values('timestamp')
                            result["Open"] = regular_sorted['o'].iloc[0]
                            result["Close"] = regular_sorted['c'].iloc[-1]
                        else:
                            result["Open"] = np.nan
                            result["Close"] = np.nan
                        for session in ['regular', 'pre', 'after']:
                            sess_group = group[group['session'] == session]
                            if not sess_group.empty:
                                result[f"{session.capitalize()} High"] = sess_group['h'].max()
                                result[f"{session.capitalize()} Low"] = sess_group['l'].min()
                            else:
                                result[f"{session.capitalize()} High"] = np.nan
                                result[f"{session.capitalize()} Low"] = np.nan
                        results.append(result)

                    daily_df = pd.DataFrame(results).sort_values('Date')
                    daily_df['Date'] = daily_df['Date'].astype(str)
                    daily_df['After Low Change (%)'] = ((daily_df['After Low'] - daily_df['Close']) / daily_df['Close']) * 100
                    daily_df['Prev Close'] = daily_df['Close'].shift(1)
                    daily_df['Pre Low Change (%)'] = ((daily_df['Pre Low'] - daily_df['Prev Close']) / daily_df['Prev Close']) * 100

                    st.subheader("Daily High/Low Prices by Session")
                    columns_to_show = [
                        'Date', 'Pre High', 'Pre Low', 'Regular High', 'Regular Low', 'Open', 'Close', 'After High', 'After Low',
                        'After Low Change (%)', 'Pre Low Change (%)'
                    ]
                    st.dataframe(daily_df[columns_to_show].style.format(
                        {col: "{:.2f}" for col in columns_to_show if col != 'Date'}, na_rep="—"
                    ), use_container_width=True)

                    st.subheader("Regular vs. Pre & After-Market Lows (Line Chart)")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=daily_df['Date'], y=daily_df['Regular Low'], name="Regular Low", mode='lines+markers'))
                    fig.add_trace(go.Scatter(x=daily_df['Date'], y=daily_df['Pre Low'], name="Pre-Market Low", mode='lines+markers'))
                    fig.add_trace(go.Scatter(x=daily_df['Date'], y=daily_df['After Low'], name="After-Hours Low", mode='lines+markers'))
                    fig.update_layout(title=f"{ticker} Low Prices by Session ({year}-{month:02d})", xaxis_title="Date", yaxis_title="Price")
                    st.plotly_chart(fig, use_container_width=True)

                    after_low_avg = daily_df['After Low Change (%)'].mean()
                    pre_low_avg = daily_df['Pre Low Change (%)'].mean()
                    st.markdown(f"""
                    **Insights:**  
                    - Average difference between after-hours low and close: {after_low_avg:.2f}%  
                    - Average difference between pre-market low and previous close: {pre_low_avg:.2f}%  
                    - Lowest daily prices often occur during the {'after-hours' if after_low_avg < pre_low_avg else 'pre-market'} session.
                    """)

else:
    st.warning("No stocks found that match your filters. Run screening to see results.")
