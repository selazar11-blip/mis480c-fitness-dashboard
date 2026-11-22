"""
=============================================================================
MIS 480C · PYTHON GROUP PROJECT
Personal Fitness & Habit Calculator Dashboard
Built with: Python, Dash, Plotly, Pandas, and Dash Bootstrap Components
=============================================================================
Research Question:
"How does your daily routine translate into calories, protein, and recovery, 
and how do you compare to the cohort?"
=============================================================================
"""

import pathlib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc

# -----------------------------------------------------------------------------
# 1. DATA LOADING & PREPARATION
# -----------------------------------------------------------------------------
# Relative path ensures smooth operation locally, in GitHub, and on Render
BASE_DIR = pathlib.Path(__file__).parent.resolve()
DATA_FILE = BASE_DIR / "fitness_tracker_dataset.csv"

# Load cohort dataset
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df["workout_type"] = df["workout_type"].fillna("Rest Day")
df["day_of_week"] = df["date"].dt.day_name()
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Available workout types in the cohort
ALL_WORKOUT_TYPES = sorted(df["workout_type"].unique().tolist())

# -----------------------------------------------------------------------------
# 2. PHYSIOLOGICAL FORMULAS, MET VALUES & GUIDELINES (DOCUMENTED DICTIONARIES)
# -----------------------------------------------------------------------------
# 1 MET (Metabolic Equivalent of Task) = 1.0 kcal / (kg of body weight * hour)
# Formula: Net Workout Energy (kcal) = (MET - 1.0) * Body_Weight_kg * (Duration_Minutes / 60)
# (MET - 1.0) is used so baseline resting metabolic rate (1.0 MET) is not counted twice alongside 24h BMR.
# For 'Rest Day', workout calories added are 0.
# Reference: Compendium of Physical Activities (Ainsworth et al.)
MET_VALUES = {
    "Running": 9.8,       # ~6 mph (10 min/mile) running = 9.8 METs
    "Cycling": 7.5,       # Moderate to vigorous cycling (12-14 mph) = 7.5 METs
    "Gym Workout": 6.0,   # Resistance training & vigorous circuit training = 6.0 METs
    "Swimming": 7.0,      # Moderate freestyle lap swimming = 7.0 METs
    "Walking": 3.8,       # Brisk walking (~3.5 mph) = 3.8 METs
    "Yoga": 3.0,          # Active Hatha / Vinyasa yoga practice = 3.0 METs
    "Rest Day": 1.2,      # Light sedentary recovery = 1.2 METs
}

# Protein recommendations (grams per kg of body weight) based on fitness goal
# Reference: International Society of Sports Nutrition (ISSN) & ACSM guidelines
PROTEIN_GUIDELINES = {
    "Maintain Fitness": {
        "multiplier": 1.4,        # Middle of 1.2 - 1.6 g/kg
        "range_str": "1.2 - 1.6 g/kg",
        "rationale": "Sustains lean mass, cellular repair, and positive nitrogen balance.",
    },
    "Build Muscle": {
        "multiplier": 2.0,        # Middle of 1.6 - 2.2 g/kg
        "range_str": "1.6 - 2.2 g/kg",
        "rationale": "Optimizes muscle protein synthesis (MPS) for progressive hypertrophy.",
    },
    "Lose Fat / Lean Out": {
        "multiplier": 2.2,        # High end: 1.8 - 2.4 g/kg
        "range_str": "1.8 - 2.4 g/kg",
        "rationale": "Maximizes satiety and prevents muscle catabolism in a caloric deficit.",
    },
}

# Default Calculator Settings
DEFAULT_WEIGHT_KG = 75
DEFAULT_WEIGHT_LBS = 165
DEFAULT_STEPS = 10000
DEFAULT_WORKOUT = "Running"
DEFAULT_WORKOUT_MINS = 45
DEFAULT_SLEEP_HOURS = 7.5
DEFAULT_GOAL = "Maintain Fitness"

# Helper for correct English ordinal suffixes (1st, 2nd, 3rd, 4th, 11th-13th, etc.)
def get_ordinal_suffix(n: float) -> str:
    """Returns correct English ordinal representation (e.g. 1st, 2nd, 3rd, 11th, 21st)."""
    val = int(round(n))
    if 11 <= (val % 100) <= 13:
        return f"{val}th"
    last = val % 10
    if last == 1:
        return f"{val}st"
    elif last == 2:
        return f"{val}nd"
    elif last == 3:
        return f"{val}rd"
    return f"{val}th"

# -----------------------------------------------------------------------------
# 3. DASH INITIALIZATION & STYLING CONSTANTS
# -----------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="MIS 480C · Personal Fitness Calculator",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# Crucial for Render deployment: Expose underlying Flask WSGI server
server = app.server

# Consistent UI Color Palette
PRIMARY_COLOR = "#0d9488"   # Solid Teal
DARK_TEAL = "#042f2e"       # Highlight Dark Teal
SECONDARY_COLOR = "#3b82f6" # Royal Blue
ACCENT_COLOR = "#f59e0b"    # Warm Amber
CORAL_COLOR = "#ef4444"     # Vibrant Coral
DARK_TEXT = "#0f172a"       # Slate 900
CAPTION_COLOR = "#334155"   # Slate 700 (darker for readability)
MUTED_TEXT = "#64748b"      # Slate 500
BG_COLOR = "#f8fafc"        # Slate 50 background
CARD_BG = "#ffffff"

CARD_STYLE = {
    "backgroundColor": CARD_BG,
    "borderRadius": "12px",
    "border": "1px solid #e2e8f0",
    "boxShadow": "0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03)",
    "marginBottom": "20px",
    "padding": "16px",
}

def make_calc_ban(card_id, sub_id, title, icon, default_val, default_sub, border_color):
    """Generates a responsive Big Awesome Number (BAN) card for calculator outputs."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Span(icon, style={"fontSize": "1.75rem", "marginRight": "10px"}),
                        html.Span(
                            title,
                            style={
                                "fontSize": "0.78rem",
                                "fontWeight": "700",
                                "color": MUTED_TEXT,
                                "textTransform": "uppercase",
                                "letterSpacing": "0.5px",
                            },
                        ),
                    ],
                    style={"display": "flex", "alignItems": "center", "marginBottom": "6px"},
                ),
                html.H3(
                    default_val,
                    id=card_id,
                    style={"fontWeight": "800", "color": DARK_TEXT, "margin": "0 0 4px 0"},
                ),
                html.Div(
                    default_sub,
                    id=sub_id,
                    style={"color": MUTED_TEXT, "fontSize": "0.78rem"},
                ),
            ]
        ),
        style={
            **CARD_STYLE,
            "borderTop": f"4px solid {border_color}",
            "marginBottom": "0px",
            "height": "100%",
        },
    )

def make_section_heading(title: str):
    """Generates a clean section heading."""
    return html.Div(
        [
            html.Span(
                title.upper(),
                style={
                    "fontSize": "0.80rem",
                    "fontWeight": "800",
                    "letterSpacing": "1.5px",
                    "color": PRIMARY_COLOR,
                    "textTransform": "uppercase",
                },
            ),
            html.Hr(style={"margin": "4px 0 16px 0", "borderColor": "#cbd5e1"}),
        ],
        style={"marginTop": "8px"},
    )

# -----------------------------------------------------------------------------
# 4. DASHBOARD LAYOUT
# -----------------------------------------------------------------------------
app.layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": BG_COLOR, "minHeight": "100vh", "padding": "20px 24px"},
    children=[
        # Hidden browser storage to preserve clicked bar state across callbacks
        dcc.Store(id="clicked-workout-store", data=None),

        # ---------------------------------------------------------------------
        # HEADER & RESEARCH QUESTION
        # ---------------------------------------------------------------------
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.Span(
                                    "MIS 480C · PYTHON GROUP PROJECT",
                                    style={"fontSize": "0.78rem", "fontWeight": "800", "color": PRIMARY_COLOR, "letterSpacing": "1.5px"},
                                ),
                                html.H1(
                                    "Personal Fitness & Habit Calculator Dashboard",
                                    style={"fontWeight": "800", "color": DARK_TEXT, "marginTop": "4px", "marginBottom": "6px", "fontSize": "1.9rem"},
                                ),
                                html.P(
                                    "Simulate your personalized daily routine, compute nutritional targets, and benchmark your habits against 80,000 cohort observations.",
                                    style={"color": MUTED_TEXT, "fontSize": "1.0rem", "marginBottom": "12px"},
                                ),
                            ]
                        ),
                        # Shortened Research Question Callout
                        dbc.Alert(
                            [
                                html.Span("🎯 Research Question: ", style={"fontWeight": "700", "color": "#0f766e"}),
                                html.Span(
                                    "How does your daily routine translate into calories, protein, and recovery, and how do you compare to the cohort?",
                                    style={"color": "#115e59", "fontWeight": "500"},
                                ),
                            ],
                            color="info",
                            style={
                                "backgroundColor": "#ccfbf1",
                                "border": "1px solid #99f6e4",
                                "borderRadius": "8px",
                                "fontSize": "0.92rem",
                                "padding": "8px 16px",
                                "marginBottom": "14px",
                            },
                        ),
                        # Single-line Disclaimer Banner
                        dbc.Alert(
                            "⚠️ Estimates only, not medical advice. Energy, macronutrient, and hydration targets are computational guidelines based on MET compendiums and sports nutrition literature.",
                            color="warning",
                            className="py-2 px-3 mb-3 text-center",
                            style={"fontSize": "0.82rem", "borderRadius": "8px", "backgroundColor": "#fef3c7", "borderColor": "#fde68a", "color": "#92400e"},
                        ),
                    ],
                    width=12,
                )
            ]
        ),

        # ---------------------------------------------------------------------
        # SECTION: YOUR PLAN
        # ---------------------------------------------------------------------
        make_section_heading("Your Plan"),

        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.H5("⚙️ Personal Health & Activity Parameters", style={"fontWeight": "700", "color": DARK_TEXT, "marginBottom": "2px"}),
                            html.Span("Adjust the parameters below to simulate your personalized daily routine.", style={"color": MUTED_TEXT, "fontSize": "0.85rem"}),
                        ],
                        style={"marginBottom": "16px"},
                    ),
                    dbc.Row(
                        [
                            # Input 1: Weight & Unit Toggle
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.Label("⚖️ Body Weight", style={"fontWeight": "600", "fontSize": "0.85rem", "color": DARK_TEXT}),
                                            dbc.RadioItems(
                                                id="weight-unit-toggle",
                                                options=[
                                                    {"label": "kg", "value": "kg"},
                                                    {"label": "lbs", "value": "lbs"},
                                                ],
                                                value="kg",
                                                inline=True,
                                                style={"fontSize": "0.80rem", "fontWeight": "600"},
                                            ),
                                        ],
                                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "4px"},
                                    ),
                                    dcc.Input(
                                        id="weight-input",
                                        type="number",
                                        value=DEFAULT_WEIGHT_KG,
                                        min=30,
                                        max=450,
                                        step=1,
                                        className="form-control",
                                        style={"borderRadius": "8px", "fontSize": "0.9rem"},
                                    ),
                                ],
                                xs=12, sm=6, md=4, lg=2,
                                style={"marginBottom": "12px"},
                            ),

                            # Input 2: Workout Type Dropdown
                            dbc.Col(
                                [
                                    html.Label("🏋️ Workout Type", style={"fontWeight": "600", "fontSize": "0.85rem", "color": DARK_TEXT, "marginBottom": "4px"}),
                                    dcc.Dropdown(
                                        id="workout-type-dropdown",
                                        options=[{"label": wt, "value": wt} for wt in ALL_WORKOUT_TYPES],
                                        value=DEFAULT_WORKOUT,
                                        clearable=False,
                                        style={"fontSize": "0.9rem"},
                                    ),
                                ],
                                xs=12, sm=6, md=4, lg=2,
                                style={"marginBottom": "12px"},
                            ),

                            # Input 3: Workout Duration Slider
                            dbc.Col(
                                [
                                    html.Label(
                                        [
                                            "⏱️ Workout Duration: ",
                                            html.Span(id="workout-mins-display", style={"color": PRIMARY_COLOR, "fontWeight": "700"}),
                                        ],
                                        style={"fontWeight": "600", "fontSize": "0.85rem", "color": DARK_TEXT, "marginBottom": "4px"},
                                    ),
                                    dcc.Slider(
                                        id="workout-minutes-slider",
                                        min=0,
                                        max=180,
                                        step=5,
                                        value=DEFAULT_WORKOUT_MINS,
                                        marks={0: "0m", 45: "45m", 90: "90m", 135: "135m", 180: "180m"},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                    ),
                                ],
                                xs=12, sm=6, md=4, lg=3,
                                style={"marginBottom": "12px"},
                            ),

                            # Input 4: Daily Steps Slider
                            dbc.Col(
                                [
                                    html.Label(
                                        [
                                            "👟 Daily Steps: ",
                                            html.Span(id="steps-display", style={"color": PRIMARY_COLOR, "fontWeight": "700"}),
                                        ],
                                        style={"fontWeight": "600", "fontSize": "0.85rem", "color": DARK_TEXT, "marginBottom": "4px"},
                                    ),
                                    dcc.Slider(
                                        id="steps-slider",
                                        min=0,
                                        max=30000,
                                        step=500,
                                        value=DEFAULT_STEPS,
                                        marks={0: "0", 10000: "10k", 20000: "20k", 30000: "30k"},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                    ),
                                ],
                                xs=12, sm=6, md=4, lg=3,
                                style={"marginBottom": "12px"},
                            ),

                            # Input 5: Sleep Hours Slider
                            dbc.Col(
                                [
                                    html.Label(
                                        [
                                            "😴 Sleep Hours: ",
                                            html.Span(id="sleep-display", style={"color": PRIMARY_COLOR, "fontWeight": "700"}),
                                        ],
                                        style={"fontWeight": "600", "fontSize": "0.85rem", "color": DARK_TEXT, "marginBottom": "4px"},
                                    ),
                                    dcc.Slider(
                                        id="sleep-hours-slider",
                                        min=3.0,
                                        max=12.0,
                                        step=0.5,
                                        value=DEFAULT_SLEEP_HOURS,
                                        marks={3: "3h", 6: "6h", 7.5: "7.5h", 9: "9h", 12: "12h"},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                    ),
                                ],
                                xs=12, sm=6, md=4, lg=2,
                                style={"marginBottom": "12px"},
                            ),
                        ]
                    ),
                    html.Hr(style={"margin": "10px 0 14px 0", "borderColor": "#e2e8f0"}),
                    dbc.Row(
                        [
                            # Input 6: Fitness Goal Dropdown
                            dbc.Col(
                                [
                                    html.Label("🎯 Primary Fitness Goal", style={"fontWeight": "600", "fontSize": "0.85rem", "color": DARK_TEXT, "marginBottom": "4px"}),
                                    dcc.Dropdown(
                                        id="goal-dropdown",
                                        options=[
                                            {"label": "Maintain Fitness (1.2 - 1.6 g/kg protein)", "value": "Maintain Fitness"},
                                            {"label": "Build Muscle (1.6 - 2.2 g/kg protein)", "value": "Build Muscle"},
                                            {"label": "Lose Fat / Lean Out (1.8 - 2.4 g/kg protein)", "value": "Lose Fat / Lean Out"},
                                        ],
                                        value=DEFAULT_GOAL,
                                        clearable=False,
                                        style={"fontSize": "0.9rem"},
                                    ),
                                ],
                                xs=12, sm=8, md=9, lg=9,
                                style={"marginBottom": "8px"},
                            ),
                            # Action: Solid Reset Button (Clear & active appearance)
                            dbc.Col(
                                [
                                    html.Label("Actions", style={"fontWeight": "600", "fontSize": "0.85rem", "color": "transparent", "display": "block", "marginBottom": "4px"}),
                                    dbc.Button(
                                        "↺ Reset All Inputs",
                                        id="reset-btn",
                                        color="primary",
                                        className="fw-bold w-100 shadow-sm",
                                        n_clicks=0,
                                        style={"fontSize": "0.88rem", "borderRadius": "8px", "height": "38px"},
                                    ),
                                ],
                                xs=12, sm=4, md=3, lg=3,
                                style={"marginBottom": "8px"},
                            ),
                        ],
                        align="end",
                    ),
                ]
            ),
            style={**CARD_STYLE, "padding": "18px 20px"},
        ),

        # ---------------------------------------------------------------------
        # SECTION: YOUR RESULTS
        # ---------------------------------------------------------------------
        make_section_heading("Your Results"),

        # Dynamic One-Sentence Summary Banner
        html.Div(
            id="dynamic-summary-sentence",
            children="Calculating personalized routine summary...",
            style={
                "backgroundColor": "#f0fdf4",
                "border": "1px solid #bbf7d0",
                "color": "#166534",
                "borderRadius": "8px",
                "padding": "12px 18px",
                "fontWeight": "600",
                "fontSize": "0.95rem",
                "marginBottom": "16px",
                "textAlign": "center",
                "boxShadow": "0 1px 3px rgba(0,0,0,0.04)",
            },
        ),

        # Result BAN Cards
        dbc.Row(
            [
                dbc.Col(
                    make_calc_ban("ban-calories", "ban-calories-sub", "Est. Daily Calories", "🔥", "2,780 kcal", "BMR + Workout + Steps", ACCENT_COLOR),
                    xs=12, sm=6, md=6, lg=3,
                    style={"marginBottom": "16px"},
                ),
                dbc.Col(
                    make_calc_ban("ban-protein", "ban-protein-sub", "Daily Protein Target", "🥩", "150 g", "Target: 1.6 - 2.2 g/kg", SECONDARY_COLOR),
                    xs=12, sm=6, md=6, lg=3,
                    style={"marginBottom": "16px"},
                ),
                dbc.Col(
                    make_calc_ban("ban-water", "ban-water-sub", "Water Intake Est.", "💧", "3.0 L (~101 oz)", "Base + sweat compensation", PRIMARY_COLOR),
                    xs=12, sm=6, md=6, lg=3,
                    style={"marginBottom": "16px"},
                ),
                dbc.Col(
                    make_calc_ban("ban-sleep", "ban-sleep-sub", "Sleep Status", "😴", "Optimal Zone ✅", "7.5 hrs (Target: 7-9 hrs)", "#8b5cf6"),
                    xs=12, sm=6, md=6, lg=3,
                    style={"marginBottom": "16px"},
                ),
            ],
            style={"marginBottom": "8px"},
        ),

        # Horizontal Stacked Calorie Breakdown Bar
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        html.H6("📊 Daily Calorie Breakdown (BMR, Workout, Steps)", style={"fontWeight": "700", "color": DARK_TEXT, "margin": "0 0 2px 0"}),
                                        html.Span("Shows how basal metabolism, active workout, and daily steps contribute to your total estimated burn.", style={"color": MUTED_TEXT, "fontSize": "0.82rem"}),
                                    ],
                                    style={"marginBottom": "4px"},
                                ),
                                dcc.Graph(
                                    id="calorie-breakdown-bar",
                                    style={"height": "220px"},
                                    config={"displayModeBar": False, "responsive": True},
                                ),
                            ],
                            style={"padding": "16px 20px"},
                        ),
                        style={**CARD_STYLE, "marginBottom": "20px"},
                    ),
                    width=12,
                )
            ]
        ),

        # ---------------------------------------------------------------------
        # SECTION: HOW YOU COMPARE
        # ---------------------------------------------------------------------
        make_section_heading("How You Compare"),

        # Equal-Height Histogram Cards (Steps, Sleep, Calories)
        dbc.Row(
            [
                # Chart 1A: Steps Distribution Benchmark
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        html.H5("👟 Steps Cohort Distribution", style={"fontWeight": "700", "color": DARK_TEXT}),
                                        html.Span("Where your daily step goal ranks against 80,000 cohort observations.", style={"color": MUTED_TEXT, "fontSize": "0.80rem"}),
                                    ],
                                    style={"marginBottom": "6px"},
                                ),
                                dcc.Graph(
                                    id="dist-steps-chart",
                                    style={"height": "340px"},
                                    config={"displayModeBar": False, "responsive": True},
                                ),
                                html.P(
                                    "Histogram of cohort daily steps. The dashed marker indicates your selected step target and overall percentile rank.",
                                    style={"color": CAPTION_COLOR, "fontSize": "0.86rem", "marginTop": "10px", "lineHeight": "1.4"},
                                ),
                            ]
                        ),
                        className="h-100 shadow-sm",
                        style=CARD_STYLE,
                    ),
                    xs=12, lg=4,
                    className="mb-3",
                ),

                # Chart 1B: Sleep Distribution Benchmark (ONE BIN PER WHOLE HOUR)
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        html.H5("😴 Sleep Hours Cohort Distribution", style={"fontWeight": "700", "color": DARK_TEXT}),
                                        html.Span("Benchmarking your sleep against cohort habits.", style={"color": MUTED_TEXT, "fontSize": "0.80rem"}),
                                    ],
                                    style={"marginBottom": "6px"},
                                ),
                                dcc.Graph(
                                    id="dist-sleep-chart",
                                    style={"height": "340px"},
                                    config={"displayModeBar": False, "responsive": True},
                                ),
                                html.P(
                                    "Histogram of sleep duration binned into whole hours (0-13h). Dashed line indicates your sleep goal and percentile.",
                                    style={"color": CAPTION_COLOR, "fontSize": "0.86rem", "marginTop": "10px", "lineHeight": "1.4"},
                                ),
                            ]
                        ),
                        className="h-100 shadow-sm",
                        style=CARD_STYLE,
                    ),
                    xs=12, lg=4,
                    className="mb-3",
                ),

                # Chart 1C: Calories Burned Benchmark
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        html.H5("🔥 Calories Burned Benchmark", style={"fontWeight": "700", "color": DARK_TEXT}),
                                        html.Span("Your calculated daily burn compared to total cohort daily calories burned.", style={"color": MUTED_TEXT, "fontSize": "0.80rem"}),
                                    ],
                                    style={"marginBottom": "6px"},
                                ),
                                dcc.Graph(
                                    id="dist-calories-chart",
                                    style={"height": "340px"},
                                    config={"displayModeBar": False, "responsive": True},
                                ),
                                html.P(
                                    "Cohort calories burned distribution. Your estimated daily burn (BMR + workout + steps) is plotted as a live marker.",
                                    style={"color": CAPTION_COLOR, "fontSize": "0.86rem", "marginTop": "10px", "lineHeight": "1.4"},
                                ),
                            ]
                        ),
                        className="h-100 shadow-sm",
                        style=CARD_STYLE,
                    ),
                    xs=12, lg=4,
                    className="mb-3",
                ),
            ],
            style={"marginBottom": "8px"},
        ),

        # What If? Comparison Chart
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        html.H5("💡 What If? Calorie Burn Across Other Workout Types", style={"fontWeight": "700", "color": DARK_TEXT}),
                                        html.Span("Compares your total daily calorie burn if you did a different workout type for the same duration with your current weight and steps.", style={"color": MUTED_TEXT, "fontSize": "0.82rem"}),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                dcc.Graph(
                                    id="what-if-chart",
                                    style={"height": "380px"},
                                    config={"displayModeBar": False, "responsive": True},
                                ),
                                html.P(
                                    "Simulates your daily calories burned if you performed a different workout type for your specified duration at your current body weight and step count. The amber bar highlights your currently selected workout.",
                                    style={"color": CAPTION_COLOR, "fontSize": "0.86rem", "marginTop": "10px", "lineHeight": "1.4"},
                                ),
                            ]
                        ),
                        style=CARD_STYLE,
                    ),
                    width=12,
                )
            ]
        ),

        # ---------------------------------------------------------------------
        # SECTION: WHAT THE COHORT LOOKS LIKE
        # ---------------------------------------------------------------------
        make_section_heading("What the Cohort Looks Like"),

        dbc.Row(
            [
                # Left: Workout Category Comparison Bar Chart (Solid Teal, Darker Highlight for Clicked)
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        html.H5("🏋️ Cohort Average Burn by Workout Type", style={"fontWeight": "700", "color": DARK_TEXT}),
                                        html.Span(
                                            "💡 Click any bar below to isolate that workout in the adjacent scatter plot! Click the same bar twice to un-filter.",
                                            style={"color": PRIMARY_COLOR, "fontSize": "0.84rem", "fontWeight": "600"},
                                        ),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                dcc.Graph(
                                    id="workout-bar-chart",
                                    style={"height": "380px"},
                                    config={"displayModeBar": False, "responsive": True},
                                ),
                                html.P(
                                    "Average calories burned per workout type across the cohort. Clicking a bar filters the adjacent scatter plot; clicking the same bar twice un-filters it.",
                                    style={"color": CAPTION_COLOR, "fontSize": "0.86rem", "marginTop": "10px", "lineHeight": "1.4"},
                                ),
                            ]
                        ),
                        style=CARD_STYLE,
                    ),
                    xs=12, md=6,
                    className="mb-3",
                ),

                # Right: Scatter Plot with User's Personal Star Marker (Reduced to 500 sample points, lower opacity)
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        html.H5("⚡ Active Minutes vs. Calories Burned", style={"fontWeight": "700", "color": DARK_TEXT}),
                                        html.Div(id="scatter-filter-badge", style={"marginBottom": "4px"}),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                dcc.Graph(
                                    id="scatter-plot",
                                    style={"height": "420px"},
                                    config={"displayModeBar": False, "responsive": True},
                                ),
                                html.P(
                                    "Cohort scatter plot (500 sampled points) with your simulated plan plotted as a glowing red star (⭐ You) using estimated active minutes (workout duration + steps/100).",
                                    style={"color": CAPTION_COLOR, "fontSize": "0.86rem", "marginTop": "10px", "lineHeight": "1.4"},
                                ),
                            ]
                        ),
                        style=CARD_STYLE,
                    ),
                    xs=12, md=6,
                    className="mb-3",
                ),
            ]
        ),

        # Heatmap Row: Default to Day-of-Week vs. Mean Calories View
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H5("🔥 Advanced Habit & Correlation Heatmap", style={"fontWeight": "700", "color": DARK_TEXT}),
                                                html.Span("Analyze weekly routine rhythms or correlation coefficients across numeric health indicators.", style={"color": MUTED_TEXT, "fontSize": "0.85rem"}),
                                            ]
                                        ),
                                        dbc.RadioItems(
                                            id="heatmap-tab-selector",
                                            options=[
                                                {"label": "🗓️ Day-of-Week vs. Mean Calories", "value": "day_of_week"},
                                                {"label": "📊 Numeric Correlation Matrix", "value": "correlation"},
                                            ],
                                            value="day_of_week",
                                            inline=True,
                                            style={"fontWeight": "600", "fontSize": "0.88rem", "color": DARK_TEXT},
                                        ),
                                    ],
                                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "flexWrap": "wrap", "marginBottom": "10px"},
                                ),
                                dcc.Graph(
                                    id="heatmap-chart",
                                    style={"height": "460px"},
                                    config={"displayModeBar": False, "responsive": True},
                                ),
                                html.P(
                                    "Weekly routine rhythm showing mean calories across days and workout types. Differences across days and workout types are minor because cohort data is synthetic. You can also view the Pearson correlation matrix using the radio selector.",
                                    style={"color": CAPTION_COLOR, "fontSize": "0.86rem", "marginTop": "10px", "lineHeight": "1.4"},
                                ),
                            ]
                        ),
                        style=CARD_STYLE,
                    ),
                    width=12,
                )
            ]
        ),

        # ---------------------------------------------------------------------
        # FOOTER
        # ---------------------------------------------------------------------
        html.Div(
            [
                html.Hr(style={"borderColor": "#cbd5e1"}),
                html.P(
                    "MIS 480C Python Group Project | Personal Fitness & Habit Calculator Dashboard",
                    style={"textAlign": "center", "fontWeight": "600", "color": DARK_TEXT, "fontSize": "0.86rem", "margin": "0 0 4px 0"},
                ),
                html.P(
                    "Cohort Dataset Note: All cohort observations (999 users, 80,000 tracking records) are synthetic data "
                    "generated for educational modeling, statistical exploration, and interactive benchmarking.",
                    style={"textAlign": "center", "color": MUTED_TEXT, "fontSize": "0.80rem", "margin": "0"},
                ),
            ],
            style={"paddingTop": "8px", "paddingBottom": "24px"},
        ),
    ],
)

# -----------------------------------------------------------------------------
# 5. INTERACTIVE CALLBACKS
# -----------------------------------------------------------------------------

# Callback 1: State Management for Reset Button, Unit Toggle & Bar ClickData
@app.callback(
    [
        Output("weight-input", "value"),
        Output("weight-unit-toggle", "value"),
        Output("steps-slider", "value"),
        Output("workout-type-dropdown", "value"),
        Output("workout-minutes-slider", "value"),
        Output("sleep-hours-slider", "value"),
        Output("goal-dropdown", "value"),
        Output("clicked-workout-store", "data"),
        Output("workout-bar-chart", "clickData"),
    ],
    [
        Input("reset-btn", "n_clicks"),
        Input("weight-unit-toggle", "value"),
        Input("workout-bar-chart", "clickData"),
    ],
    [
        State("weight-input", "value"),
        State("clicked-workout-store", "data"),
    ],
    prevent_initial_call=True,
)
def manage_calculator_state(reset_clicks, unit, click_data, current_weight, current_clicked):
    """
    Unified state controller:
    1. If Reset Button clicked: resets all sliders and inputs to original defaults.
    2. If Unit Toggle clicked: converts weight between kg and lbs.
    3. If Workout Bar clicked: stores or toggles the clicked workout type for scatter filtering.
       Resetting clickData to None ensures subsequent clicks on the exact same bar trigger cleanly!
    """
    try:
        triggered = ctx.triggered_id
    except Exception:
        triggered = None

    # 1. Reset Button Action: Clear all filters & inputs
    if triggered == "reset-btn":
        return (
            DEFAULT_WEIGHT_KG,
            "kg",
            DEFAULT_STEPS,
            DEFAULT_WORKOUT,
            DEFAULT_WORKOUT_MINS,
            DEFAULT_SLEEP_HOURS,
            DEFAULT_GOAL,
            None,
            None,
        )

    # 2. Unit Toggle Action (kg <-> lbs conversion)
    if triggered == "weight-unit-toggle":
        if current_weight is None:
            new_weight = DEFAULT_WEIGHT_KG if unit == "kg" else DEFAULT_WEIGHT_LBS
        else:
            if unit == "lbs":
                new_weight = round(current_weight * 2.20462)
            else:
                new_weight = round(current_weight / 2.20462)
        return (
            new_weight,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

    # 3. Bar Chart Click Action: Toggle selection (click same bar twice to un-filter)
    if triggered == "workout-bar-chart":
        if not click_data or "points" not in click_data:
            raise dash.exceptions.PreventUpdate

        clicked_cat = click_data["points"][0]["x"]
        # If user clicked the same category again, toggle off (None), else set to clicked category
        new_cat = None if clicked_cat == current_clicked else clicked_cat

        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            new_cat,
            None,  # Reset clickData so the same bar can be clicked again immediately
        )

    return (
        dash.no_update,
        dash.no_update,
        dash.no_update,
        dash.no_update,
        dash.no_update,
        dash.no_update,
        dash.no_update,
        dash.no_update,
        dash.no_update,
    )


# Callback 2: Slider Label Live Updates
@app.callback(
    [
        Output("workout-mins-display", "children"),
        Output("steps-display", "children"),
        Output("sleep-display", "children"),
    ],
    [
        Input("workout-minutes-slider", "value"),
        Input("steps-slider", "value"),
        Input("sleep-hours-slider", "value"),
    ],
)
def update_slider_labels(workout_mins, steps, sleep):
    """Displays current values dynamically next to slider titles."""
    return f"{workout_mins} min", f"{steps:,} steps", f"{sleep:.1f} hrs"


# Callback 3: Calculator Callback Updating Results, BANs, Breakdown Bar, 3 Histograms & What-If Chart
@app.callback(
    [
        Output("dynamic-summary-sentence", "children"),
        Output("ban-calories", "children"),
        Output("ban-calories-sub", "children"),
        Output("ban-protein", "children"),
        Output("ban-protein-sub", "children"),
        Output("ban-water", "children"),
        Output("ban-water-sub", "children"),
        Output("ban-sleep", "children"),
        Output("ban-sleep-sub", "children"),
        Output("calorie-breakdown-bar", "figure"),
        Output("dist-steps-chart", "figure"),
        Output("dist-sleep-chart", "figure"),
        Output("dist-calories-chart", "figure"),
        Output("what-if-chart", "figure"),
    ],
    [
        Input("weight-input", "value"),
        Input("weight-unit-toggle", "value"),
        Input("steps-slider", "value"),
        Input("workout-type-dropdown", "value"),
        Input("workout-minutes-slider", "value"),
        Input("sleep-hours-slider", "value"),
        Input("goal-dropdown", "value"),
    ],
)
def update_calculator_dashboard(weight_val, unit, steps, workout_type, workout_mins, sleep_hours, goal):
    """
    Personalized Calculator Callback:
    Triggered exclusively by personal parameter adjustments. Updates:
    - Dynamic one-sentence summary
    - 4 Personalized BAN output cards
    - Horizontal stacked calorie breakdown bar (BMR, Workout, Steps)
    - 3 Cohort Distribution charts with user markers & percentiles (Steps, Sleep, Calories)
    - 'What If?' bar chart simulating other workout types
    """
    # 1. Parse & Normalize Body Weight to kg
    safe_weight = weight_val if weight_val and weight_val > 0 else 70
    weight_kg = safe_weight if unit == "kg" else safe_weight * 0.453592
    safe_steps = steps if steps is not None else 10000
    safe_mins = workout_mins if workout_mins is not None else 45
    safe_sleep = sleep_hours if sleep_hours is not None else 7.5
    safe_workout = workout_type if workout_type else "Running"
    safe_goal = goal if goal in PROTEIN_GUIDELINES else "Maintain Fitness"

    # 2. Personalized Physiological Calculations
    # (A) Daily Caloric Burn (BMR + Net Workout Burn + Steps Non-Exercise Thermogenesis)
    # Uses (MET - 1) so resting baseline isn't counted twice; Rest Day adds 0 workout calories.
    bmr = 24.0 * 1.0 * weight_kg
    if safe_workout == "Rest Day":
        workout_kcal = 0.0
    else:
        net_met = max(0.0, MET_VALUES.get(safe_workout, 1.0) - 1.0)
        workout_kcal = net_met * weight_kg * (safe_mins / 60.0)

    step_kcal = safe_steps * 0.04 * (weight_kg / 70.0)
    total_calories = round(bmr + workout_kcal + step_kcal)

    # (B) Daily Protein Target
    prot_info = PROTEIN_GUIDELINES[safe_goal]
    protein_g = round(weight_kg * prot_info["multiplier"])

    # (C) Daily Water Intake Estimate (35 mL/kg baseline + 500 mL/hr exercise sweat replenishment)
    water_liters = round((weight_kg * 35.0 + safe_mins * (500.0 / 60.0)) / 1000.0, 1)
    water_oz = round(water_liters * 33.814)

    # (D) Sleep Recovery Status
    if safe_sleep < 6.0:
        sleep_badge = "Sleep Deficit ⚠️"
        sleep_desc = f"{safe_sleep:.1f} hrs (Deficit: Elevated cortisol & fatigue)"
    elif safe_sleep < 7.0:
        sleep_badge = "Suboptimal 🟡"
        sleep_desc = f"{safe_sleep:.1f} hrs (Suboptimal: 7-9 hrs recommended)"
    elif safe_sleep <= 9.0:
        sleep_badge = "Optimal Zone ✅"
        sleep_desc = f"{safe_sleep:.1f} hrs (Prime zone for muscular & neural repair)"
    else:
        sleep_badge = "Extended Rest 💤"
        sleep_desc = f"{safe_sleep:.1f} hrs (Extended restorative sleep)"

    # BAN Card Content
    ban_cal_val = f"{total_calories:,} kcal"
    ban_cal_sub = f"BMR: {bmr:,.0f} | Workout: +{workout_kcal:,.0f} | Steps: +{step_kcal:,.0f}"

    ban_prot_val = f"{protein_g} grams"
    ban_prot_sub = f"Target: {prot_info['range_str']} ({safe_goal})"

    ban_water_val = f"{water_liters} L"
    ban_water_sub = f"~{water_oz} fl oz (Base + sweat compensation)"

    ban_sleep_val = sleep_badge
    ban_sleep_sub = sleep_desc

    # 3. Calculate Cohort Percentiles
    user_active_mins = round(safe_mins + (safe_steps / 100.0), 1)
    pct_steps = (df["steps"] < safe_steps).mean() * 100.0
    pct_sleep = (df["sleep_hours"] < safe_sleep).mean() * 100.0
    pct_cal = (df["calories_burned"] < total_calories).mean() * 100.0

    # Dynamic Summary Sentence matching the steps percentile displayed below
    summary_sentence = (
        f"✨ {safe_mins} min of {safe_workout} plus {safe_steps:,} steps burns about "
        f"{total_calories:,} kcal, and your step count beats {pct_steps:.0f}% of the cohort."
    )

    # 4. Horizontal Calorie Breakdown Bar
    # Ordered BMR, Workout, Steps left to right to match the bar segments
    fig_breakdown = go.Figure()
    fig_breakdown.add_trace(
        go.Bar(
            y=["Daily Calories"],
            x=[bmr],
            name="BMR",
            orientation="h",
            marker={"color": "#475569"},
            text=[f"BMR: {bmr:,.0f} kcal"],
            textposition="inside",
            hovertemplate="<b>Resting Metabolic Baseline (BMR)</b><br>Basal Energy: <b>%{x:,.0f} kcal</b><extra></extra>",
        )
    )
    fig_breakdown.add_trace(
        go.Bar(
            y=["Daily Calories"],
            x=[workout_kcal],
            name="Workout",
            orientation="h",
            marker={"color": "#f59e0b"},
            text=[f"Workout: {workout_kcal:,.0f} kcal"],
            textposition="inside",
            hovertemplate=f"<b>Workout Activity ({safe_workout})</b><br>Exercise Burn: <b>%{{x:,.0f}} kcal</b><extra></extra>",
        )
    )
    fig_breakdown.add_trace(
        go.Bar(
            y=["Daily Calories"],
            x=[step_kcal],
            name="Steps",
            orientation="h",
            marker={"color": PRIMARY_COLOR},
            text=[f"Steps: {step_kcal:,.0f} kcal"],
            textposition="inside",
            hovertemplate="<b>Steps Activity</b><br>Steps Burn: <b>%{x:,.0f} kcal</b><extra></extra>",
        )
    )
    fig_breakdown.update_layout(
        title={"text": f"Daily Calorie Breakdown — Total: {total_calories:,.0f} kcal", "font": {"size": 14, "color": DARK_TEXT}},
        barmode="stack",
        height=220,
        autosize=True,
        margin={"l": 20, "r": 20, "t": 40, "b": 90},
        xaxis={"title": "Calories (kcal)", "showgrid": True, "gridcolor": "#f1f5f9"},
        yaxis={"showticklabels": False},
        legend={
            "orientation": "h",
            "y": -0.45,
            "x": 0,
            "yanchor": "top",
            "xanchor": "left",
            "traceorder": "normal",
            "font": {"size": 11},
        },
        plot_bgcolor=CARD_BG,
        paper_bgcolor=CARD_BG,
    )

    # 5. Build Distribution Chart 1: Steps Benchmark
    fig_steps = go.Figure()
    fig_steps.add_trace(
        go.Histogram(
            x=df["steps"],
            nbinsx=35,
            marker={"color": "#94a3b8", "line": {"width": 0.5, "color": "#ffffff"}},
            opacity=0.75,
            hovertemplate="Steps: %{x:,}<br>Cohort Count: %{y:,}<extra></extra>",
        )
    )
    fig_steps.add_vline(
        x=safe_steps,
        line_width=3,
        line_dash="dash",
        line_color=PRIMARY_COLOR,
        annotation_text=f"📍 You: {safe_steps:,} ({get_ordinal_suffix(pct_steps)} pct)",
        annotation_position="top right",
        annotation_font={"size": 11, "color": PRIMARY_COLOR, "weight": "bold"},
    )
    fig_steps.update_layout(
        title={"text": f"You're in the {get_ordinal_suffix(pct_steps)} percentile for steps", "font": {"size": 13, "color": DARK_TEXT}},
        plot_bgcolor=CARD_BG,
        paper_bgcolor=CARD_BG,
        height=340,
        autosize=True,
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
        xaxis={"title": "Daily Steps", "showgrid": True, "gridcolor": "#f1f5f9"},
        yaxis={"title": "Count", "showgrid": True, "gridcolor": "#f1f5f9"},
        showlegend=False,
    )

    # 6. Build Distribution Chart 2: Sleep Benchmark (ONE BIN PER WHOLE HOUR)
    fig_sleep = go.Figure()
    fig_sleep.add_trace(
        go.Histogram(
            x=df["sleep_hours"],
            xbins={"start": 0, "end": 13, "size": 1},
            autobinx=False,
            marker={"color": "#a78bfa", "line": {"width": 1, "color": "#ffffff"}},
            opacity=0.75,
            hovertemplate="Sleep Bin: [%{x}, %{x}+1) hrs<br>Cohort Count: %{y:,}<extra></extra>",
        )
    )
    fig_sleep.add_vline(
        x=safe_sleep,
        line_width=3,
        line_dash="dash",
        line_color="#7c3aed",
        annotation_text=f"📍 You: {safe_sleep:.1f}h ({get_ordinal_suffix(pct_sleep)} pct)",
        annotation_position="top right",
        annotation_font={"size": 11, "color": "#7c3aed", "weight": "bold"},
    )
    fig_sleep.update_layout(
        title={"text": f"You're in the {get_ordinal_suffix(pct_sleep)} percentile for sleep", "font": {"size": 13, "color": DARK_TEXT}},
        plot_bgcolor=CARD_BG,
        paper_bgcolor=CARD_BG,
        height=340,
        autosize=True,
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
        xaxis={"title": "Sleep Duration (hours)", "showgrid": True, "gridcolor": "#f1f5f9", "dtick": 1},
        yaxis={"title": "Count", "showgrid": True, "gridcolor": "#f1f5f9"},
        showlegend=False,
    )

    # 7. Build Distribution Chart 3: Calories Burned Benchmark
    fig_cal = go.Figure()
    fig_cal.add_trace(
        go.Histogram(
            x=df["calories_burned"],
            nbinsx=35,
            marker={"color": "#fbbf24", "line": {"width": 0.5, "color": "#ffffff"}},
            opacity=0.75,
            hovertemplate="Calories: %{x:,.0f} kcal<br>Cohort Count: %{y:,}<extra></extra>",
        )
    )
    fig_cal.add_vline(
        x=total_calories,
        line_width=3,
        line_dash="dash",
        line_color=ACCENT_COLOR,
        annotation_text=f"📍 You: {total_calories:,} kcal ({get_ordinal_suffix(pct_cal)} pct)",
        annotation_position="top right",
        annotation_font={"size": 11, "color": "#b45309", "weight": "bold"},
    )
    fig_cal.update_layout(
        title={"text": f"You're in the {get_ordinal_suffix(pct_cal)} percentile for calories burned", "font": {"size": 13, "color": DARK_TEXT}},
        plot_bgcolor=CARD_BG,
        paper_bgcolor=CARD_BG,
        height=340,
        autosize=True,
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
        xaxis={"title": "Calories Burned (kcal)", "showgrid": True, "gridcolor": "#f1f5f9"},
        yaxis={"title": "Count", "showgrid": True, "gridcolor": "#f1f5f9"},
        showlegend=False,
    )

    # 8. Build "What If?" Comparison Bar Chart (using (MET - 1) for net calories, Rest Day = 0)
    what_if_rows = []
    for w_name, met_val in MET_VALUES.items():
        if w_name == "Rest Day":
            w_burn = round(bmr + step_kcal)
        else:
            net_m = max(0.0, met_val - 1.0)
            w_burn = round(bmr + net_m * weight_kg * (safe_mins / 60.0) + step_kcal)
        what_if_rows.append({"workout": w_name, "calories": w_burn, "is_current": (w_name == safe_workout)})
    what_if_df = pd.DataFrame(what_if_rows).sort_values(by="calories", ascending=True)

    # Highlight currently selected workout in amber (#f59e0b), other workouts in solid teal (#0d9488)
    what_if_colors = ["#f59e0b" if r["is_current"] else PRIMARY_COLOR for _, r in what_if_df.iterrows()]

    max_what_if = what_if_df["calories"].max() if not what_if_df.empty else 3000
    x_max_what_if = max_what_if * 1.15

    fig_what_if = go.Figure(
        go.Bar(
            y=what_if_df["workout"],
            x=what_if_df["calories"],
            orientation="h",
            text=what_if_df["calories"].apply(lambda v: f"{v:,.0f} kcal"),
            textposition="outside",
            cliponaxis=False,
            marker_color=what_if_colors,
            marker_line_width=1,
            marker_line_color="#e2e8f0",
            hovertemplate="<b>%{y}</b><br>Estimated Daily Burn: <b>%{x:,.0f} kcal</b><extra></extra>",
        )
    )
    fig_what_if.update_layout(
        title={"text": f"What If? Daily Burn Across Workout Types ({safe_mins} min workout)", "font": {"size": 13, "color": DARK_TEXT}},
        plot_bgcolor=CARD_BG,
        paper_bgcolor=CARD_BG,
        height=380,
        autosize=True,
        margin={"l": 100, "r": 40, "t": 40, "b": 50},
        xaxis={"title": "Estimated Calories Burned (kcal)", "range": [0, x_max_what_if], "showgrid": True, "gridcolor": "#f1f5f9"},
        yaxis={"title": "Workout Type", "showgrid": False},
        showlegend=False,
    )

    return (
        summary_sentence,
        ban_cal_val,
        ban_cal_sub,
        ban_prot_val,
        ban_prot_sub,
        ban_water_val,
        ban_water_sub,
        ban_sleep_val,
        ban_sleep_sub,
        fig_breakdown,
        fig_steps,
        fig_sleep,
        fig_cal,
        fig_what_if,
    )


# Callback 4: Dedicated Callback for Cohort Bar Chart & Heatmap
# Bar chart highlights clicked bar in dark teal (#042f2e) and all other bars in solid teal (#0d9488)
@app.callback(
    [
        Output("workout-bar-chart", "figure"),
        Output("heatmap-chart", "figure"),
    ],
    [
        Input("heatmap-tab-selector", "value"),
        Input("clicked-workout-store", "data"),
    ],
)
def update_cohort_reference_charts(heatmap_mode, clicked_workout):
    """
    Renders cohort reference charts:
    - Workout Category Comparison Bar Chart (solid teal with darker shade for clicked bar)
    - Correlation or Day-of-Week Activity Heatmap (defaults to Day-of-Week vs. Mean Calories)
    """
    # 1. Build Workout Category Comparison Bar Chart
    bar_data = (
        df.groupby("workout_type")["calories_burned"]
        .mean()
        .reset_index()
        .sort_values(by="calories_burned", ascending=False)
    )

    # Use solid teal (#0d9488) for all bars; highlight only clicked bar in darker shade (#042f2e)
    bar_colors = [
        DARK_TEAL if (clicked_workout and row["workout_type"] == clicked_workout) else PRIMARY_COLOR
        for _, row in bar_data.iterrows()
    ]

    fig_bar = go.Figure(
        go.Bar(
            x=bar_data["workout_type"],
            y=bar_data["calories_burned"],
            text=bar_data["calories_burned"].round(0).apply(lambda x: f"{x:,.0f} kcal"),
            textposition="outside",
            marker_color=bar_colors,
            marker_line_width=1,
            marker_line_color="#e2e8f0",
            hovertemplate="<b>%{x}</b><br>Cohort Mean: <b>%{y:,.0f} kcal</b><br><i>Click bar to filter/un-filter scatter plot</i><extra></extra>",
        )
    )
    fig_bar.update_layout(
        title={"text": "Cohort Average Calories Burned by Workout Type", "font": {"size": 14, "color": DARK_TEXT}},
        plot_bgcolor=CARD_BG,
        paper_bgcolor=CARD_BG,
        height=380,
        autosize=True,
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
        xaxis={"title": "Workout Type", "showgrid": False},
        yaxis={"title": "Mean Calories Burned (kcal)", "showgrid": True, "gridcolor": "#f1f5f9"},
    )

    # 2. Build Advanced Heatmap (Defaults to Day-of-Week vs. Mean Calories)
    if heatmap_mode == "correlation":
        numeric_cols = ["steps", "calories_burned", "distance_km", "active_minutes", "sleep_hours", "heart_rate_avg"]
        col_clean = ["Steps", "Calories Burned", "Distance (km)", "Active Min", "Sleep (hrs)", "Heart Rate"]
        corr = df[numeric_cols].corr().round(2)
        corr.columns = col_clean
        corr.index = col_clean

        fig_heat = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Tealrose",
            zmin=-1.0,
            zmax=1.0,
        )
        fig_heat.update_traces(
            hovertemplate="X: <b>%{x}</b><br>Y: <b>%{y}</b><br>Pearson r: <b>%{z:.2f}</b><extra></extra>"
        )
        fig_heat.update_layout(
            title={"text": "Correlation Matrix of Numeric Habits & Physical Indicators", "font": {"size": 14, "color": DARK_TEXT}},
            plot_bgcolor=CARD_BG,
            paper_bgcolor=CARD_BG,
            height=460,
            autosize=True,
            margin={"l": 70, "r": 20, "t": 50, "b": 60},
            coloraxis_colorbar={"title": "Pearson r", "thickness": 14},
        )
    else:
        pivot_data = (
            df.pivot_table(
                index="workout_type",
                columns="day_of_week",
                values="calories_burned",
                aggfunc="mean",
            )
            .reindex(columns=DAY_ORDER)
            .round(0)
        )
        fig_heat = px.imshow(
            pivot_data,
            text_auto=",.0f",
            aspect="auto",
            color_continuous_scale="Viridis",
            labels={"x": "Day of Week", "y": "Workout", "color": "Mean Calories (kcal)"},
        )
        fig_heat.update_traces(
            hovertemplate="Day: <b>%{x}</b><br>Workout: <b>%{y}</b><br>Mean Burn: <b>%{z:,.0f} kcal</b><extra></extra>"
        )
        fig_heat.update_layout(
            title={"text": "Weekly Routine Rhythm: Mean Calories across Days & Workout Types", "font": {"size": 14, "color": DARK_TEXT}},
            plot_bgcolor=CARD_BG,
            paper_bgcolor=CARD_BG,
            height=460,
            autosize=True,
            margin={"l": 70, "r": 20, "t": 50, "b": 60},
            xaxis_title="Day of Week",
            yaxis_title="Workout",
            coloraxis_colorbar={"title": "kcal", "thickness": 14},
        )

    return fig_bar, fig_heat


# Callback 5: Updates Scatter Plot with Personal Plan Star Marker & Bar Click Filter
# Star marker uses estimated active minutes (workout minutes + steps/100) with top right label
@app.callback(
    [
        Output("scatter-plot", "figure"),
        Output("scatter-filter-badge", "children"),
    ],
    [
        Input("weight-input", "value"),
        Input("weight-unit-toggle", "value"),
        Input("steps-slider", "value"),
        Input("workout-type-dropdown", "value"),
        Input("workout-minutes-slider", "value"),
        Input("clicked-workout-store", "data"),
    ],
)
def update_scatter_with_user_marker(weight_val, unit, steps, workout_type, workout_mins, clicked_workout):
    """
    Renders the cohort scatter plot (Active Minutes vs. Calories Burned).
    - Reduced sample of 500 points with lowered opacity (0.45) for clean visuals.
    - User star plotted at x = workout minutes + steps / 100 with textposition='top right'.
    - Removed duplicate legend star and renamed legend title to 'Workout'.
    """
    safe_weight = weight_val if weight_val and weight_val > 0 else 70
    weight_kg = safe_weight if unit == "kg" else safe_weight * 0.453592
    safe_steps = steps if steps is not None else 10000
    safe_mins = workout_mins if workout_mins is not None else 45
    safe_workout = workout_type if workout_type else "Running"

    # Compute personal plan calories (using net MET (MET - 1.0) so resting baseline isn't double-counted; Rest Day = 0)
    bmr = 24.0 * 1.0 * weight_kg
    if safe_workout == "Rest Day":
        workout_kcal = 0.0
    else:
        net_met = max(0.0, MET_VALUES.get(safe_workout, 1.0) - 1.0)
        workout_kcal = net_met * weight_kg * (safe_mins / 60.0)
    user_cal = round(bmr + workout_kcal + safe_steps * 0.04 * (weight_kg / 70.0))

    # User's estimated active minutes: workout duration + (steps / 100)
    user_active_mins = round(safe_mins + (safe_steps / 100.0), 1)

    # Determine filter condition from bar clickData
    plot_df = df
    is_filtered = False
    if clicked_workout and clicked_workout in df["workout_type"].values:
        plot_df = df[df["workout_type"] == clicked_workout]
        is_filtered = True
        badge = html.Div(
            [
                dbc.Badge(
                    f"🎯 Filtered by Bar Click: {clicked_workout}",
                    color="primary",
                    style={"fontSize": "0.82rem", "padding": "5px 10px", "marginRight": "8px"},
                ),
                html.Span("(Click bar again to un-filter)", style={"color": MUTED_TEXT, "fontSize": "0.80rem"}),
            ],
            style={"display": "flex", "alignItems": "center"},
        )
    else:
        badge = html.Span("Showing cohort distribution (sample of 500 points)", style={"color": MUTED_TEXT, "fontSize": "0.80rem"})

    # Sample exactly 500 cohort points for a cleaner, less noisy scatter plot
    scatter_sample = plot_df.sample(min(500, len(plot_df)), random_state=42)

    color_map = {
        "Running": "#ef4444",
        "Cycling": "#f59e0b",
        "Gym Workout": "#3b82f6",
        "Swimming": "#06b6d4",
        "Walking": "#10b981",
        "Yoga": "#8b5cf6",
        "Rest Day": "#94a3b8",
    }

    fig_scatter = px.scatter(
        scatter_sample,
        x="active_minutes",
        y="calories_burned",
        color="workout_type",
        color_discrete_map=color_map,
        opacity=0.45,
        custom_data=["workout_type", "heart_rate_avg", "sleep_hours", "steps"],
        labels={"workout_type": "Workout"},
    )
    fig_scatter.update_traces(
        marker={"size": 6, "line": {"width": 0.4, "color": "#ffffff"}},
        hovertemplate=(
            "<b>Workout: %{customdata[0]}</b><br>"
            "Active: <b>%{x} min</b><br>"
            "Calories Burned: <b>%{y:,.0f} kcal</b><br>"
            "Heart Rate: <b>%{customdata[1]} bpm</b><br>"
            "Steps: <b>%{customdata[3]:,}</b>"
            "<extra></extra>"
        ),
    )

    # Add the User's Personal Marker: textposition='top right', showlegend=False
    fig_scatter.add_trace(
        go.Scatter(
            x=[user_active_mins],
            y=[user_cal],
            mode="markers+text",
            marker={
                "size": 18,
                "color": "#dc2626",
                "symbol": "star",
                "line": {"width": 2, "color": "#ffffff"},
            },
            text=["You (Current Plan)"],
            textposition="top right",
            textfont={"size": 11, "color": "#dc2626", "weight": "bold"},
            showlegend=False,
            hovertemplate=(
                "<b>Your Simulated Plan</b><br>"
                f"Workout Type: <b>{safe_workout}</b><br>"
                f"Est. Active Minutes: <b>{user_active_mins:.0f} min</b> (workout: {safe_mins}m + steps/100: {round(safe_steps/100)}m)<br>"
                f"Est. Daily Calories: <b>{user_cal:,} kcal</b><br>"
                f"Daily Steps: <b>{safe_steps:,}</b>"
                "<extra></extra>"
            ),
        )
    )

    # Ensure x-axis range is comfortably padded so the user star is always clearly visible
    max_active_cohort = scatter_sample["active_minutes"].max() if not scatter_sample.empty else 1400
    x_max = max(max_active_cohort, user_active_mins) + 60

    fig_scatter.update_layout(
        title={
            "text": f"Active Minutes vs. Calories Burned ({'Single Workout: ' + clicked_workout if is_filtered else 'All Workouts'})",
            "font": {"size": 14, "color": DARK_TEXT},
        },
        plot_bgcolor=CARD_BG,
        paper_bgcolor=CARD_BG,
        height=420,
        autosize=True,
        margin={"l": 50, "r": 20, "t": 40, "b": 80},
        xaxis={
            "title": "Active Minutes (Daily: Workout + Steps/100)",
            "range": [0, x_max],
            "showgrid": True,
            "gridcolor": "#f1f5f9",
        },
        yaxis={"title": "Calories Burned (kcal)", "showgrid": True, "gridcolor": "#f1f5f9"},
        legend={"title": {"text": "Workout"}, "orientation": "h", "yanchor": "bottom", "y": -0.35, "xanchor": "center", "x": 0.5},
    )

    return fig_scatter, badge


# -----------------------------------------------------------------------------
# 6. LOCAL ENTRYPOINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting MIS 480C Personal Fitness Calculator Dashboard on http://127.0.0.1:8050 ...")
    app.run(debug=True, host="127.0.0.1", port=8050)
