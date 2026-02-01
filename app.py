import streamlit as st

st.set_page_config(page_title="Monte Carlo Trading Suite", layout="wide")

st.title("📊 Monte Carlo Trading Suite")

st.markdown("""
### Welcome to the Trading Strategy Simulation Suite

This application allows you to simulate your trading strategies using Monte Carlo methods to better understand potential outcomes, drawdowns, and risk management.
""")

st.markdown("#### Available Tools:")

st.page_link("pages/2_Stress_Test.py", label="**Stress Test**", icon="🧪")
st.markdown("""
Advanced simulation using non-linear probability distributions. Model fat-tailed Reward:Risk ratios (outliers) and Win Rates following a Beta distribution to stress test your strategy under more realistic market conditions.
""")

st.markdown("---")
st.markdown("**<- Select the Stress Test tool from the sidebar to get started!**")

st.info("💡 Tip: Use the Stress Test to see how even a small probability of large rewards or losses can affect your long-term equity curve.")
