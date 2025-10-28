import streamlit as st
import pandas as pd
import joblib as jl

model = jl.load("fraud_detection_pipeline.pkl")
