import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from simulation import run_monte_carlo, get_beta_params, sample_beta_dist, get_lognormal_params, sample_lognormal_dist, lognormal_clipped_mean, get_cond_mean_bounds

st.set_page_config(page_title="Stress Test - Monte Carlo", layout="wide")

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

st.title("🧪 Strategy Stress Test")
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
risk_per_trade = st.sidebar.number_input("Risk per Trade (%)", min_value=0.0, max_value=100.0, value=1.0) / 100.0
num_sims = 1000

# Distribution Configuration
col_dist1, col_dist2 = st.columns(2)

with col_dist1:
    with st.container(border=True):
        st.subheader("Win Rate")
        st.markdown("🎲 **Model:** Beta Distribution")
        st.info("Model win rate variability across paths. Rule: p ~ Beta(α, β)")
        
        wr_avg = st.number_input("Average Win Rate (%)", value=28.0, help="This is the long-term average win rate you expect. Internally we use a Beta distribution because it is perfect for modeling probabilities (0–100%) that vary over time.") / 100.0
        wr_vol = st.number_input("Win Rate Std Dev (%)", value=6.0, help="Controls how stable or unstable your win rate is across different market conditions. Higher value = more variation between good and bad periods → larger drawdowns possible. We use Beta distribution to keep values realistic between ~5–40%.") / 100.0
        
        wr_min_p = st.number_input("Min Plausible Win Rate (%)", value=22.0, help="The lowest win rate you think is realistically possible (even in very bad markets). Helps prevent the model from generating unrealistically low values.") / 100.0
        wr_max_p = st.number_input("Max Plausible Win Rate (%)", value=33.0, help="The highest win rate you believe the system can achieve in favorable conditions. Used to shape the tails of the Beta distribution.") / 100.0
        
        st.caption("This section uses a Beta distribution because win rates are proportions (0–1) that naturally vary and stay bounded. It captures regime changes (bad markets ~18%, good ~30%) much better than a fixed percentage. [Learn more](https://distribution-explorer.github.io/continuous/beta.html)")
        
        wr_alpha, wr_beta = get_beta_params(wr_avg, wr_vol)
        
        if wr_alpha is None:
            st.error(f"⚠️ Impossible Std Dev: {wr_vol*100:.1f}% for mean {wr_avg*100:.1f}%. Max allowed: {np.sqrt(wr_avg*(1-wr_avg))*100:.1f}%")
            st.stop()

with col_dist2:
    with st.container(border=True):
        st.subheader("Reward:Risk")
        st.markdown("📈 **Model:** Log-Normal Distribution")
        st.info("Outlier Capture Model (Fat Tail).")
        
        def_median, def_mean, def_prob10, def_max = 3.0, 3.38, 0.10, 60.0

        st.caption("This section uses a Log-Normal distribution because it naturally creates fat right tails — perfect for systems where a small % of trades deliver very large payoffs and drive most of the profit.")
        
        rr_median = st.number_input("Median R:R (Typical value)", value=def_median, help="The 'middle' Reward:Risk you see in most winning trades. For outlier systems, this is usually low (2–4×).")
        rr_mean_cond = st.number_input("Average R:R (excluding outliers > 10:1)", value=def_mean, min_value=1.1, max_value=9.9, help="The average size of your 'normal' (non-outlier) winning trades. Must be less than 10. The system will automatically add the big outliers on top of this based on the percentage you provide below.")
        
        rr_tail_help = """
        This is a percentage (%).  
        Enter the approximate % of ALL trades (winners + losers) that you realistically expect to have a Reward:Risk ratio of 10:1 or better.  
        Example: If you set 15%, it means roughly 15 out of every 100 trades should be big winners paying at least 10 times your risk (R:R ≥ 10).  
        The other ~85% of trades will have smaller payoffs (including all losers and small/medium winners).  

        **Why this matters:**  
        - **5–10%** → more balanced system, less explosive  
        - **10–18%** → typical for good outlier-capture systems (most common sweet spot)  
        - **18–25%** → very aggressive, high dependence on rare big winners  
        - **>25%** → extreme asymmetry, huge variance in results  

        In real outlier strategies, this number often falls between 10% and 20%.  
        If you have backtest data, count: (number of trades with profit ≥ 10 × risk) ÷ (total trades) × 100
        """
        
        rr_prob10_raw = st.number_input(
            "How fat are the tails? (% of trades with R:R ≥ 10:1)", 
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
        
        st.caption("This section uses a Log-Normal distribution because it naturally creates fat right tails — exactly what happens in outlier-capture systems. [Learn more](https://distribution-explorer.github.io/continuous/lognormal.html)")
        
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
        
        st.metric("Final Simulated Average R:R", f"{simulated_mean:.2f}", help=f"""
        This is the actual average R:R used in the simulation. 
        
        **How it's calculated:**
        1. **Model Fit:** The system finds a Log-Normal distribution that perfectly fits your "Median R:R", your "Average for normal trades (<10:1)", and your "Outlier %".
        2. **Clipping:** Since outliers can theoretically be infinite in a Log-Normal model, we 'clip' (cap) values at your "Maximum realistic R:R" ({rr_max_cap}).
        3. **Expected Value:** This metric is the mathematical average of that capped distribution.
        
        *Note: Without the cap, the mathematical average would be {total_theo_mean:.2f}.*
        """)

        if total_theo_mean > simulated_mean * 1.5:
            st.warning(f"⚠️ **Heavy Clipping:** The distribution's tail is much fatter than your Cap ({rr_max_cap}). You are losing significant expectancy ({total_theo_mean - simulated_mean:.2f} R) due to the cap. Consider increasing the Maximum realistic R:R if you believe larger winners are possible.")

st.markdown("---")
if st.button("🚀 Run Stress Test", type="primary", use_container_width=True):
    st.rerun()
st.markdown("---")

# Visualization of Distributions
# Sampling for preview
preview_size = 10000
wr_samples = sample_beta_dist(wr_alpha, wr_beta, preview_size)
rr_samples = sample_lognormal_dist(rr_mu, rr_sigma, preview_size, clip_min=0.1, clip_max=rr_max_cap)

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
    wr_vector = sample_beta_dist(wr_alpha, wr_beta, num_sims)
    
    # Reward:Risk is still dynamic per trade within each path
    rr_matrix = sample_lognormal_dist(rr_mu, rr_sigma, (num_sims, trades_per_sim), clip_min=0.1, clip_max=rr_max_cap)
    
    results = run_monte_carlo(start_balance, trades_per_sim, wr_vector, rr_matrix, risk_per_trade, num_sims)
    
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
        - **Max Drawdown (95th):** {max_dd_95:.1f}%
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
        - **Percentiles:** 50%: {p50_rr:.1f} | 75%: {p75_rr:.1f} | 90%: {p90_rr:.1f} | 95%: {p95_rr:.1f}
        - **Probabilities:** R:R>10: {prob_gt10*100:.1f}% | R:R>20: {prob_gt20*100:.1f}% | R:R>50: {prob_gt50*100:.1f}%
        - **Theoretical Mean:** {total_theo_mean:.2f}
        - **Simulated Mean:** {np.mean(rr_samples):.2f}
        """)

# RUN SIMULATION
st.markdown("---")

median_path = np.median(results, axis=0)
x = np.arange(trades_per_sim + 1)

fig = go.Figure()
for i in range(num_sims):
    fig.add_trace(go.Scatter(
        x=x, y=results[i],
        mode='lines',
        line=dict(width=0.5, color='rgba(150, 150, 150, 0.1)'),
        hoverinfo='skip', showlegend=False
    ))
    
fig.add_trace(go.Scatter(
    x=x, y=median_path,
    mode='lines', line=dict(width=3, color='#007bff'),
    name='Median Curve'
))

# Calculate scale to be "centered" around paths (excluding extreme outliers)
all_final_values = results[:, -1]
y_min = np.percentile(results, 5) # 5th percentile of all points
y_max = np.percentile(results, 95) # 95th percentile of all points

# Ensure start balance and median are always visible
y_min = min(y_min, start_balance * 0.8)
y_max = max(y_max, median_path[-1] * 1.2)

fig.update_layout(
    xaxis_title="Trade Number", yaxis_title="Balance ($)",
    template="plotly_white", hovermode="x unified",
    height=550,
    margin=dict(l=60, r=20, t=10, b=50),
    yaxis=dict(range=[y_min, y_max]),
    showlegend=True,
    legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.99, bgcolor="rgba(255, 255, 255, 0.5)")
)

# Add Title as Annotation inside the chart
fig.add_annotation(
    xref="paper", yref="paper",
    x=0.01, y=0.99,
    text=f"<b>Equity Growth</b><br><span style='font-size: 12px; color: #666'>(Final Median: ${median_path[-1]:,.2f})</span>",
    showarrow=False,
    font=dict(size=18, color="#333"),
    align="left",
    xanchor="left", yanchor="top"
)

# Add annotation for final median value inside the chart
fig.add_annotation(
    x=trades_per_sim,
    y=median_path[-1],
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
