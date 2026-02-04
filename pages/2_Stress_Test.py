import streamlit as st
import numpy as np
import pandas as pd
import io
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy.stats import norm, beta
from simulation import run_monte_carlo, get_beta_params, sample_beta_dist, get_lognormal_params, sample_lognormal_dist, lognormal_clipped_mean, get_cond_mean_bounds

st.set_page_config(page_title="Tradesystem Stress Test - Monte Carlo", layout="wide")

# Custom CSS for light theme
st.markdown("""
<style>
    .stNumberInput { margin-bottom: 10px; }
    .metric-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value { font-size: 20px; font-weight: bold; color: #007bff; }
    .metric-label { font-size: 12px; color: #6c757d; }
    .percentile-info {
        font-size: 12px;
        color: #495057;
        margin-top: 2px;
        line-height: 1.2;
    }
    .chart-container-pdf {
        display: flex;
        flex-direction: column;
        padding: 0px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧪 Tradesystem Stress Test")
st.markdown("Test your strategy against non-linear distributions and outliers.")

def get_gamma_params(mean, std, loc):
    """
    Converts Mean, Std Dev, and Shift (Loc) to Gamma Shape and Scale.
    """
    adj_mean = max(0.0001, mean - loc)
    safe_std = max(0.0001, std)
    scale = (safe_std ** 2) / adj_mean
    shape = adj_mean / scale
    return shape, scale

# Sidebar - General Params
st.sidebar.header("General Parameters")
start_balance = st.sidebar.number_input("Starting Balance ($)", min_value=0.0, value=10000.0, step=1000.0, format="%.2f")
trades_per_sim = st.sidebar.number_input("Trades per Simulation", min_value=1, max_value=5000, value=50)
risk_per_trade = st.sidebar.number_input("Risk per Trade (%)", min_value=0.0, max_value=100.0, value=0.30) / 100.0
# num_sims moved to UI section below

# Distribution Configuration
col_dist1, col_dist2 = st.columns(2)

with col_dist1:
    with st.container(border=True):
        st.subheader("Win Rate")
        st.markdown("🎲 **Model:** Beta Distribution")
        st.info("Model win rate variability across paths. Rule: p ~ Beta(α, β)")
        
        wr_avg = st.number_input("Average Win Rate (%)", value=28.0, help="This is the long-term average win rate you expect (including breakeven trades). Internally we use a Beta distribution because it is perfect for modeling probabilities (0–100%) that vary over time.") / 100.0
        wr_vol = st.number_input("Win Rate Std Dev (%)", value=6.0, help="Controls how stable or unstable your win rate is across different market conditions. Higher value = more variation between good and bad periods → larger drawdowns possible. We use Beta distribution to keep values realistic between ~5–40%.") / 100.0
        
        wr_min_p = st.number_input("Min Plausible Win Rate (%)", value=14.3, help="The lowest win rate you think is realistically possible. The simulation will actively clip (limit) any sampled win rate to this minimum value to ensure realism.") / 100.0
        wr_max_p = st.number_input("Max Plausible Win Rate (%)", value=44.7, help="The highest win rate you believe the system can achieve. The simulation will actively clip (limit) any sampled win rate to this maximum value.") / 100.0
        
        st.caption("This section uses a Beta distribution because win rates are proportions (0–1) that naturally vary and stay bounded. It captures regime changes (e.g. bad markets ~18%, good ~30%) much better than a fixed percentage. [Learn more](https://distribution-explorer.github.io/continuous/beta.html)")
        
        wr_alpha, wr_beta = get_beta_params(wr_avg, wr_vol)
        
        if wr_alpha is None:
            st.error(f"⚠️ Impossible Std Dev: {wr_vol*100:.1f}% for mean {wr_avg*100:.1f}%. Max allowed: {np.sqrt(wr_avg*(1-wr_avg))*100:.1f}%")
            st.stop()
        
        # Calculate naturally occurring tails (e.g. 0.5% and 99.5%)
        suggested_min = beta.ppf(0.005, wr_alpha, wr_beta)
        suggested_max = beta.ppf(0.995, wr_alpha, wr_beta)
        
        st.markdown(f"""
        <div style="background-color: #e7f3ff; padding: 10px; border-radius: 5px; border-left: 5px solid #007bff; margin-top: 20px; margin-bottom: 15px;">
            <p style="margin-bottom: 5px; font-weight: bold; font-size: 13px; color: #0056b3;">💡 Symmetrical Suggestion</p>
            <p style="margin: 0; font-size: 12px; color: #333;">To avoid probability "spikes" at the edges, try using these bounds which match the distribution's natural tails:</p>
            <p style="margin-top: 5px; font-weight: bold; font-size: 12px; color: #0056b3;">Min: {suggested_min*100:.1f}% | Max: {suggested_max*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

with col_dist2:
    with st.container(border=True):
        st.subheader("Reward:Risk")
        st.markdown("📈 **Model:** Log-Normal Distribution")
        st.info("Outlier Capture Model (Fat Tail).")
        
        def_median, def_mean, def_prob10, def_max = 5.0, 5.17, 0.10, 60.0


        rr_median = st.number_input("Median R:R of Wins (Typical value)", value=def_median, help="The 'middle' Reward:Risk you see in most winning trades. For outlier systems, this is usually low (2–4×).")
        rr_mean_cond = st.number_input("Average R:R of Wins (Normal range < 10:1)", value=def_mean, min_value=1.1, max_value=9.9, help="The average size of your 'normal' (non-outlier) winning trades. Must be less than 10. The system will automatically add the big outliers on top of this based on the percentage you provide below.")
        
        rr_tail_help = """
        This is a percentage (%).  
        Enter the approximate % of WINNING trades that you realistically expect to have a Reward:Risk ratio of 10:1 or better.  
        Example: If you set 15%, it means roughly 15 out of every 100 winning trades should be big winners paying at least 10 times your risk (R:R ≥ 10).  

        **Why this matters:**  
        - **5–10%** of wins → more balanced system, less explosive  
        - **10–18%** of wins → typical for good outlier-capture systems (most common sweet spot)  
        - **18–25%** of wins → very aggressive, high dependence on rare big winners  
        - **>25%** of wins → extreme asymmetry, huge variance in results  

        In real outlier strategies, this number often falls between 10% and 20% of winning trades.  
        If you have backtest data, count: (number of trades with profit ≥ 10 × risk) ÷ (total number of winning trades) × 100
        """
        
        rr_prob10_raw = st.number_input(
            "Outlier Win Frequency (% of WINNING trades ≥ 10:1)", 
            min_value=0.1, 
            max_value=60.0, 
            value=float(def_prob10 * 100), 
            step=0.5,
            help=rr_tail_help
        )
        
        if rr_prob10_raw > 25.0:
            st.warning("⚠️ High Tail Fatness (>25%): This assumes an extremely aggressive system with huge variance.")
        elif rr_prob10_raw < 5.0:
            st.info("ℹ️ Low Tail Fatness (<5%): This assumes a more conservative system with fewer large outliers.")

        rr_prob10 = rr_prob10_raw / 100.0
        rr_max_cap = st.number_input("Maximum realistic R:R (cap)", value=def_max, help="The largest R:R you consider realistic (e.g. 50×, 100×, 200×). We clip extreme values to avoid simulation instability, but still allow fat tails.")
        
        st.caption("This section uses a Log-Normal distribution because it naturally creates fat right tails — exactly what happens in outlier-capture systems where a small % of trades deliver very large payoffs and drive most of the profit. [Learn more](https://distribution-explorer.github.io/continuous/lognormal.html)")
        
        # Check Math Limits for the conditional mean
        min_limit, max_limit = get_cond_mean_bounds(np.log(rr_median), 10)
        
        if rr_mean_cond > max_limit:
            st.error(f"⚠️ **Too High:** For a Median of {rr_median}, the highest possible average for normal trades (<10:1) is **{max_limit:.2f}**. The system will use {max_limit:.2f} instead.")
            rr_mean_actual_input = max_limit
        elif rr_mean_cond < min_limit:
            # If Median=5 and Input=2, min_limit is probably around 5.0. 
            # Values below median are possible but require huge variance which isn't what the user likely wants.
            st.error(f"⚠️ **Too Low:** For a Median of {rr_median}, the typical average for normal trades (<10:1) should be at least **{min_limit:.2f}**. Your input of {rr_mean_cond} is mathematically inconsistent. The system will adjust parameters to get as close as possible.")
            rr_mean_actual_input = rr_mean_cond # Solver will handle it by increasing sigma
        else:
            rr_mean_actual_input = rr_mean_cond

        rr_mu, rr_sigma = get_lognormal_params(rr_median, rr_mean_actual_input, rr_prob10)
        
        # Theoretical Total Mean (Unclipped)
        total_theo_mean = np.exp(rr_mu + (rr_sigma**2 / 2))
        
        # Simulated Total Mean (Clipped at rr_max_cap)
        simulated_mean = lognormal_clipped_mean(rr_mu, rr_sigma, rr_max_cap)
        
        # Display metric with custom HTML to allow side-by-side uncapped value
        st.markdown(f"""
        <div class="metric-container" style="text-align: left; padding: 10px 15px;">
            <div class="metric-label" style="margin-bottom: 2px;">Final Simulated Average R:R <span style="cursor:help;" title="This is the actual average R:R used in the simulation after applying the {rr_max_cap} cap.">ℹ️</span></div>
            <div style="display: flex; align-items: baseline; gap: 12px;">
                <div class="metric-value" style="font-size: 28px;">{simulated_mean:.2f}</div>
                <div style="font-size: 14px; color: #6c757d; font-weight: normal;">
                    (Uncapped: {total_theo_mean:.2f})
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if total_theo_mean > simulated_mean * 1.5:
            st.info(f"ℹ️ **Model Fitting Notice:** To match your Outlier % and Median, the system is using a distribution with a very fat right tail. Since you have chosen to cap winners at **{rr_max_cap}:1** for realism, the simulation will ignore the 'infinite' mathematical tail of the Log-Normal model. This ensures your results base themselves on your realistic expectations rather than extreme statistical outliers.")

st.markdown("---")
num_sims = st.selectbox("Number of Simulations", options=[1000, 2000, 5000], index=0, help="Higher numbers provide more stable statistical data but take slightly longer to process.")

if st.button("🚀 Run Stress Test", type="primary", use_container_width=True):
    st.rerun()
st.markdown("---")

# Visualization of Distributions
# Sampling for preview
preview_size = 10000
wr_samples = sample_beta_dist(wr_alpha, wr_beta, preview_size, clip_min=wr_min_p, clip_max=wr_max_p)
rr_samples = sample_lognormal_dist(rr_mu, rr_sigma, preview_size, clip_min=0.00001, clip_max=rr_max_cap)

# Statistical Calculations (Preview)
p5 = np.percentile(wr_samples, 5)
p50 = np.percentile(wr_samples, 50)
p95 = np.percentile(wr_samples, 95)
prob_out = (np.sum(wr_samples < wr_min_p) + np.sum(wr_samples > wr_max_p)) / preview_size

p50_rr = np.percentile(rr_samples, 50)
p75_rr = np.percentile(rr_samples, 75)
p90_rr = np.percentile(rr_samples, 90)
p95_rr = np.percentile(rr_samples, 95)
p99_rr = np.percentile(rr_samples, 99)
prob_gt10 = np.mean(rr_samples > 10)
prob_gt20 = np.mean(rr_samples > 20)
prob_gt50 = np.mean(rr_samples > 50)

st.subheader("Previewing Distributions (PDF)")

prev_col1, prev_col2 = st.columns(2)

with prev_col1:
    with st.container(border=True):
        st.markdown('<div class="chart-container-pdf">', unsafe_allow_html=True)
        st.markdown("<b>Win Rate PDF</b>", unsafe_allow_html=True)
        if np.var(wr_samples) > 1e-9:
            fig_wr = ff.create_distplot([wr_samples], ['Win Rate (Beta)'], bin_size=0.01, show_hist=False, colors=['#007bff'])
            fig_wr.update_layout(template="plotly_white", height=220, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_wr, use_container_width=True, config={'displayModeBar': False})
        else:
            st.warning("Win Rate volatility too low to generate density curve.")
        
        st.markdown('</div>', unsafe_allow_html=True)

with prev_col2:
    with st.container(border=True):
        st.markdown('<div class="chart-container-pdf">', unsafe_allow_html=True)
        st.markdown("<b>R:R PDF (Outliers)</b>", unsafe_allow_html=True)
        if np.var(rr_samples) > 1e-9:
            fig_rr = ff.create_distplot([rr_samples], ['Reward:Risk'], bin_size=0.1, show_hist=False, colors=['#28a745'])
            fig_rr.update_layout(template="plotly_white", height=220, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_rr, use_container_width=True, config={'displayModeBar': False})
        else:
            st.warning("R:R volatility too low to generate density curve.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# RUN SIMULATION (Moved up to gather metrics for Summary)
with st.spinner("Simulating..."):
    # Rule 8: Draw ONE value p ~ Beta(α, β) for each path
    wr_vector = sample_beta_dist(wr_alpha, wr_beta, num_sims, clip_min=wr_min_p, clip_max=wr_max_p)
    
    # Reward:Risk is still dynamic per trade within each path
    rr_matrix = sample_lognormal_dist(rr_mu, rr_sigma, (num_sims, trades_per_sim), clip_min=0.00001, clip_max=rr_max_cap)
    
    results, outcomes_matrix = run_monte_carlo(start_balance, trades_per_sim, wr_vector, rr_matrix, risk_per_trade, num_sims)
    
    # PERFORMANCE CALCULATIONS
    final_balances = results[:, -1]
    median_final = np.median(final_balances)
    equity_growth = ((median_final / start_balance) - 1) * 100
    
    # Max Drawdowns per path
    drawdowns = []
    for i in range(num_sims):
        path = results[i]
        peaks = np.maximum.accumulate(path)
        dd = (peaks - path) / peaks
        drawdowns.append(np.max(dd))
    max_dd_95 = np.percentile(drawdowns, 95) * 100
    max_dd_std = np.std(drawdowns) * 100
    
    # Consecutive Losing Streaks (Median of Max Streaks)
    # We need to simulate the trades again or check PnL
    # Since run_monte_carlo returns balance, we find PnL differences
    max_streaks = []
    for i in range(num_sims):
        path = results[i]
        pnl = np.diff(path)
        # 1 for loss, 0 for win
        is_loss = (pnl < 0).astype(int)
        # Count consecutive 1s
        count = 0
        max_c = 0
        for val in is_loss:
            if val == 1:
                count += 1
                max_c = max(max_c, count)
            else:
                count = 0
        max_streaks.append(max_c)
    median_streak = np.median(max_streaks)
    mean_streak = np.mean(max_streaks)
    p95_streak = np.percentile(max_streaks, 95)
    
    win_sims = np.sum(final_balances > start_balance)
    ruined_paths = np.any(results < (start_balance * 0.6), axis=1)
    bankruptcy = np.sum(ruined_paths)

# CONSOLIDATED SUMMARY SECTION
st.markdown("---")
st.subheader("📊 Detailed Distribution Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    with st.container(border=True):
        st.markdown("##### Simulation Performance")
        st.markdown(f"""
        - **Median Equity Growth:** {equity_growth:+.1f}%
        - **Max Drawdown (95th):** {max_dd_95:.1f}% | Std Dev: {max_dd_std:.1f}%
        - **Max Loss Streak:** Median: {median_streak:.0f} | 95th: {p95_streak:.0f}
        - **Total Trades per Session:** {trades_per_sim}
        - **Profit Probability:** {(win_sims/num_sims)*100:.1f}%
        """)

with summary_col2:
    with st.container(border=True):
        st.markdown("##### Win Rate (Beta)")
        st.markdown(f"""
        - **5th Percentile:** {p5*100:.1f}%
        - **Median (50th):** {p50*100:.1f}%
        - **95th Percentile:** {p95*100:.1f}%
        - **Probability outside [{wr_min_p*100:.0f}%, {wr_max_p*100:.0f}%]:** {prob_out*100:.1f}%
        """)

with summary_col3:
    with st.container(border=True):
        st.markdown("##### Reward:Risk (Log-Normal)")
        st.markdown(f"""
        - **Percentiles (Wins):** 50%: {p50_rr:.1f} | 75%: {p75_rr:.1f} | 90%: {p90_rr:.1f} | 95%: {p95_rr:.1f}
        - **Win Probabilities:** R:R>10: {prob_gt10*100:.1f}% | R:R>20: {prob_gt20*100:.1f}% | R:R>50: {prob_gt50*100:.1f}%
        - **Theoretical Mean:** {total_theo_mean:.2f}
        - **Simulated Mean:** {np.mean(rr_samples):.2f}
        """)

# RUN SIMULATION VISUALIZATION
median_path = np.median(results, axis=0)
x = np.arange(trades_per_sim + 1)
all_final_values = results[:, -1]

# Create subplots: 1 row, 2 columns (Equity Paths + Distribution)
fig = make_subplots(
    rows=1, cols=2, 
    shared_yaxes=True, 
    column_widths=[0.85, 0.15],
    horizontal_spacing=0.01,
    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
)

# 1. Main Equity Paths (Left)
for i in range(num_sims):
    fig.add_trace(go.Scatter(
        x=x, y=results[i],
        mode='lines',
        line=dict(width=0.5, color='rgba(150, 150, 150, 0.1)'),
        hoverinfo='skip', showlegend=False
    ), row=1, col=1)
    
fig.add_trace(go.Scatter(
    x=x, y=median_path,
    mode='lines', line=dict(width=3, color='#007bff'),
    name='Median Curve'
), row=1, col=1)

# 2. Final Equity Distribution (Right / 90 degrees)
fig.add_trace(go.Histogram(
    y=all_final_values,
    name='Final Distribution',
    marker_color='rgba(0, 123, 255, 0.8)',
    orientation='h',
    nbinsy=100, # Increased granularity
    showlegend=False,
    hovertemplate='Balance: %{y:,.0f}<br>Paths: %{x}<extra></extra>'
), row=1, col=2)

# Calculate scale to be "centered" around paths (excluding extreme outliers)
y_min = np.percentile(results, 5) # 5th percentile of all points
y_max = np.percentile(results, 95) # 95th percentile of all points

# Ensure start balance and median are always visible
y_min = min(y_min, start_balance * 0.8)
y_max = max(y_max, median_path[-1] * 1.25)

fig.update_layout(
    template="plotly_white", 
    hovermode="x unified",
    height=550,
    margin=dict(l=60, r=20, t=10, b=50),
    showlegend=True,
    legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.84, bgcolor="rgba(255, 255, 255, 0.5)")
)

# Update X-Axes
fig.update_xaxes(title_text="Trade Number", row=1, col=1)
fig.update_xaxes(title_text="Density", showticklabels=False, row=1, col=2)

# Update Y-Axes (Shared)
fig.update_yaxes(title_text="Balance ($)", range=[y_min, y_max], row=1, col=1)
fig.update_yaxes(showgrid=False, row=1, col=2)

# Add Title as Annotation inside the chart
fig.add_annotation(
    xref="paper", yref="paper",
    x=0.01, y=0.99,
    text=f"<b>Equity Growth & Distribution</b><br><span style='font-size: 12px; color: #666'>(Final Median: ${median_path[-1]:,.2f})</span>",
    showarrow=False,
    font=dict(size=18, color="#333"),
    align="left",
    xanchor="left", yanchor="top"
)

# Add annotation for final median value inside the chart
fig.add_annotation(
    x=trades_per_sim,
    y=median_path[-1],
    xref="x1", yref="y1",
    text=f" <b>${median_path[-1]:,.2f}</b>",
    showarrow=True,
    arrowhead=2,
    ax=-50, ay=0,
    xanchor="right",
    font=dict(color="#007bff", size=12),
    bgcolor="rgba(255, 255, 255, 0.9)",
    bordercolor="#007bff",
    borderwidth=1,
    borderpad=4
)

st.markdown("---")
with st.container(border=True):
    st.plotly_chart(fig, use_container_width=True)

# Summary (Moved metrics calculation up, just showing Ruin here)
s_col1, s_col2 = st.columns(2)
with s_col1:
    # Profit Probability already in summary_col3 above
    pass
with s_col2:
    if bankruptcy > 0:
        st.error(f"Risk of Ruin (40% DD): **{(bankruptcy/num_sims)*100:.1f}%**")
    else:
        st.info("No simulation reached ruin (based on 40% drawdown).")

# --- NEW SECTION: DRAW SETS OF R:R ---
st.markdown("---")
st.subheader("🎯 Sampled Trade Sets (R:R Sequences)")

# Pick 10 random indices for sampling sets
num_sampled_sets = 10
sampled_indices = np.random.choice(num_sims, size=num_sampled_sets, replace=False)

sampled_data = []
for i, idx in enumerate(sampled_indices):
    rr_seq = rr_matrix[idx]
    path = results[idx]
    outcomes = outcomes_matrix[idx]
    
    # Calculate metrics for this specific set
    total_ret = ((path[-1] / start_balance) - 1) * 100
    
    set_win_rate = (np.sum(outcomes == 1) / trades_per_sim) * 100
    set_rr_avg = np.mean(rr_seq[outcomes == 1]) if np.any(outcomes == 1) else 0.0
    
    # Create a display sequence where losses are shown as -1
    display_rr_seq = np.where(outcomes == 1, rr_seq, -1.0)
    
    sampled_data.append({
        "Set": f"Set {i+1}",
        "Total Return (%)": f"{total_ret:+.2f}%",
        "Win Average (%)": f"{set_win_rate:.1f}%",
        "Avg Win R:R": f"{set_rr_avg:.2f}",
        "RR_Sequence": display_rr_seq,
        "Outcomes": outcomes,
        "Raw_Return": total_ret,
        "Raw_WinRate": set_win_rate,
        "Raw_RR": set_rr_avg
    })

# R:R Sequence Chart
fig_sets = go.Figure()
colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

for i, data in enumerate(sampled_data):
    fig_sets.add_trace(go.Scatter(
        x=np.arange(1, trades_per_sim + 1),
        y=data["RR_Sequence"],
        mode='lines+markers',
        name=data["Set"],
        line=dict(color=colors[i], width=1),
        marker=dict(size=4),
        hovertemplate="Trade %{x}<br>R:R: %{y:.2f}<extra></extra>"
    ))

fig_sets.update_layout(
    template="plotly_white",
    height=400,
    margin=dict(l=60, r=20, t=20, b=50),
    xaxis_title="Trade Number",
    yaxis_title="Reward:Risk Ratio",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

with st.container(border=True):
    st.plotly_chart(fig_sets, use_container_width=True)

# Summary Table
table_df = pd.DataFrame(sampled_data)[["Set", "Total Return (%)", "Win Average (%)", "Avg Win R:R"]]
st.markdown("##### Sampled Sets Summary")
st.dataframe(table_df.set_index("Set").T, use_container_width=True)

# Excel Export
def to_excel(sampled_data):
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Summary
            summary_df = pd.DataFrame(sampled_data)[["Set", "Total Return (%)", "Win Average (%)", "Avg Win R:R"]]
            summary_df.to_excel(writer, index=False, sheet_name='Summary')
            
            # Sheet 2: All sequences
            seq_dict = {"Trade": np.arange(1, trades_per_sim + 1)}
            for data in sampled_data:
                seq_dict[f"{data['Set']} R:R"] = data["RR_Sequence"]
                seq_dict[f"{data['Set']} Result"] = ["Win" if o == 1 else "Loss" for o in data["Outcomes"]]
            
            seq_df = pd.DataFrame(seq_dict)
            seq_df.to_excel(writer, index=False, sheet_name='Trade Sequences')
            
        return output.getvalue()
    except Exception as e:
        st.error(f"Excel Export failed: {str(e)}")
        return None

excel_data = to_excel(sampled_data)
st.download_button(
    label="📥 Export Sets to Excel",
    data=excel_data,
    file_name="monte_carlo_sampled_sets.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True
)
