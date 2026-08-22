import streamlit as st
import pandas as pd
import joblib

from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DepressionRiskAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 19px;
        opacity: 0.75;
        margin-top: 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_FILE = (
    PROJECT_ROOT
    / "ml"
    / "final_model.joblib"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


try:

    model = load_model()

except Exception as e:

    st.error(
        "Unable to load the trained AI model."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# PHQ-9 OPTIONS
# ============================================================

options = [
    "Not at all",
    "Several days",
    "More than half the days",
    "Nearly every day"
]


# ============================================================
# PHQ-9 SCORING
# ============================================================

score_map = {

    "Not at all": 0,

    "Several days": 1,

    "More than half the days": 2,

    "Nearly every day": 3
}


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<p class="main-title">🧠 DepressionRiskAI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">AI-Powered Depression Risk Screening</p>',
    unsafe_allow_html=True
)

st.write(
    """
    A machine-learning research prototype that analyzes
    PHQ-9 questionnaire responses and estimates a
    depression-severity category.
    """
)

st.warning(
    "⚠️ Research and educational screening tool only. "
    "This application does not provide a medical diagnosis."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🧠 DepressionRiskAI")

    st.write(
        """
        This project combines the PHQ-9 questionnaire
        with machine learning to estimate depression
        severity.
        """
    )

    st.divider()

    st.subheader("🤖 Model")

    st.write("Algorithm: Linear SVM")
    st.write("Kernel: Linear")
    st.write("C: 100")

    st.divider()

    st.subheader("📊 Research Dataset")

    st.write("Records: 682")
    st.write("Input Columns: 16")
    st.write("Severity Classes: 5")

    st.divider()

    st.subheader("📈 Research Performance")

    st.write("Accuracy: 91.97%")
    st.write("Precision: 92.18%")
    st.write("Recall: 91.97%")
    st.write("Weighted F1: 91.92%")
    st.write("Macro F1: 91.12%")

    st.divider()

    st.caption(
        "Final Year Project — AI-Based Depression "
        "Risk Screening System"
    )


# ============================================================
# MAIN LAYOUT
# ============================================================

left, right = st.columns([2, 1])


# ============================================================
# USER INFORMATION
# ============================================================

with right:

    st.subheader("👤 User Information")

    age = st.number_input(
        "Age",
        min_value=10,
        max_value=100,
        value=20
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    sleep_quality = st.selectbox(
        "Sleep Quality",
        [
            "Good",
            "Average",
            "Bad",
            "Worst"
        ]
    )

    study_pressure = st.selectbox(
        "Study Pressure",
        [
            "Good",
            "Average",
            "Bad",
            "Worst"
        ]
    )

    financial_pressure = st.selectbox(
        "Financial Pressure",
        [
            "Good",
            "Average",
            "Bad",
            "Worst"
        ]
    )


# ============================================================
# PHQ-9 QUESTIONS
# ============================================================

with left:

    st.subheader("📝 PHQ-9 Assessment")

    st.write(
        "Over the last two weeks, how often have you "
        "experienced the following?"
    )

    st.divider()

    q1 = st.selectbox(
        "1. Little interest or pleasure in doing things",
        options,
        key="q1"
    )

    q2 = st.selectbox(
        "2. Feeling down, depressed, or hopeless",
        options,
        key="q2"
    )

    q3 = st.selectbox(
        "3. Trouble falling or staying asleep, or sleeping too much",
        options,
        key="q3"
    )

    q4 = st.selectbox(
        "4. Feeling tired or having little energy",
        options,
        key="q4"
    )

    q5 = st.selectbox(
        "5. Poor appetite or overeating",
        options,
        key="q5"
    )

    q6 = st.selectbox(
        "6. Feeling bad about yourself—or that you are a failure "
        "or have let yourself or your family down",
        options,
        key="q6"
    )

    q7 = st.selectbox(
        "7. Trouble concentrating on things, such as reading "
        "or watching television",
        options,
        key="q7"
    )

    q8 = st.selectbox(
        "8. Moving or speaking so slowly that other people "
        "could have noticed? Or being unusually restless",
        options,
        key="q8"
    )

    q9 = st.selectbox(
        "9. Thoughts that you would be better off dead "
        "or of hurting yourself in some way",
        options,
        key="q9"
    )


# ============================================================
# PHQ-9 ANSWERS
# ============================================================

answers = [
    q1,
    q2,
    q3,
    q4,
    q5,
    q6,
    q7,
    q8,
    q9
]


# ============================================================
# PHQ-9 SCORE
# ============================================================

phq_score = sum(
    score_map[answer]
    for answer in answers
)


# ============================================================
# PHQ-9 CATEGORY
# ============================================================

if phq_score <= 4:

    phq_category = "Minimal"

elif phq_score <= 9:

    phq_category = "Mild"

elif phq_score <= 14:

    phq_category = "Moderate"

elif phq_score <= 19:

    phq_category = "Moderately severe"

else:

    phq_category = "Severe"


# ============================================================
# ASSESSMENT PROGRESS
# ============================================================

st.progress(
    1.0,
    text="PHQ-9 assessment: 9/9 questions ready"
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.divider()

analyze = st.button(
    "🔍 Analyze Depression Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    # ========================================================
    # MODEL INPUT
    # ========================================================

    input_data = pd.DataFrame([
        {

            "Age":
                age,

            "Gender":
                gender,

            "Little interest or pleasure in doing things":
                q1,

            "Feeling down, depressed, or hopeless":
                q2,

            "Trouble falling or staying asleep, or sleeping too much":
                q3,

            "Feeling tired or having little energy":
                q4,

            "Poor appetite or overeating":
                q5,

            "Feeling bad about yourself—or that you are a failure or have let yourself or your family down":
                q6,

            "Trouble concentrating on things, such as reading the newspaper or watching television":
                q7,

            "Moving or speaking so slowly that other people could have noticed? Or the opposite—being so fidgety or restless that you have been moving around a lot more than usual":
                q8,

            "Thoughts that you would be better off dead or of hurting yourself in some way":
                q9,

            "Sleep Quality":
                sleep_quality,

            "Study Pressure":
                study_pressure,

            "Financial Pressure":
                financial_pressure

        }
    ])


    # ========================================================
    # MACHINE LEARNING PREDICTION
    # ========================================================

    try:

        prediction = model.predict(
            input_data
        )[0]

        probabilities = model.predict_proba(
            input_data
        )[0]

        classes = model.classes_

        prediction_index = list(classes).index(
            prediction
        )

        confidence = probabilities[
            prediction_index
        ]

    except Exception as e:

        st.error(
            "An error occurred while generating "
            "the prediction."
        )

        st.code(str(e))

        st.stop()


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.header("📊 Assessment Results")


    # ========================================================
    # RESULT METRICS
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "PHQ-9 Score",
            f"{phq_score}/27"
        )

    with col2:

        st.metric(
            "PHQ-9 Category",
            phq_category
        )

    with col3:

        st.metric(
            "AI Prediction",
            prediction
        )

        st.caption(
            "Machine-learning screening result"
        )


    # ========================================================
    # MODEL PREDICTION PROBABILITY
    # ========================================================

    st.subheader(
        "🎯 Model Prediction Probability"
    )

    st.progress(
        min(float(confidence), 1.0),
        text=f"{confidence * 100:.1f}%"
    )

    st.caption(
        """
        This percentage represents the model's predicted
        probability for the selected severity category.
        It should NOT be interpreted as clinical certainty.
        """
    )


    if confidence >= 0.90:

        st.info(
            """
            The model assigns a high probability to this
            predicted category. This does not mean that the
            prediction is clinically certain.
            """
        )

    elif confidence >= 0.60:

        st.info(
            """
            The model shows moderate-to-high probability
            for this predicted category.
            """
        )

    else:

        st.warning(
            """
            The model probability is relatively low,
            indicating greater uncertainty in the prediction.
            """
        )


    # ========================================================
    # PROBABILITY DISTRIBUTION
    # ========================================================

    st.subheader(
        "📈 Prediction Distribution"
    )

    probability_df = pd.DataFrame(
        {
            "Severity":
                classes,

            "Probability (%)":
                probabilities * 100
        }
    )

    probability_df[
        "Probability (%)"
    ] = (
        probability_df[
            "Probability (%)"
        ].round(2)
    )

    chart_df = probability_df.set_index(
        "Severity"
    )

    st.bar_chart(
        chart_df,
        y="Probability (%)"
    )

    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # PHQ-9 VS AI
    # ========================================================

    st.subheader(
        "🔬 PHQ-9 vs AI"
    )

    comparison_df = pd.DataFrame(
        {
            "Method": [
                "Standard PHQ-9",
                "Machine Learning"
            ],

            "Result": [
                phq_category,
                prediction
            ]
        }
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # PHQ-9 SAFETY CHECK
    # ========================================================

    if phq_score >= 20:

        st.error(
            """
            ⚠️ The PHQ-9 score falls in the severe range.

            This research prototype cannot assess your safety
            or provide a diagnosis. Please consider seeking
            appropriate evaluation from a qualified mental-health
            professional.
            """
        )

    elif phq_score >= 15:

        st.warning(
            """
            The PHQ-9 score falls in the moderately severe range.

            Consider discussing your symptoms with a qualified
            mental-health professional.
            """
        )


    # ========================================================
    # QUESTION 9 SAFETY CHECK
    # ========================================================

    if q9 != "Not at all":

        st.error(
            """
            ⚠️ Important

            You selected an answer other than "Not at all"
            for the question concerning thoughts of being
            better off dead or hurting yourself.

            This application cannot determine your immediate
            safety or provide an emergency assessment.

            Please consider seeking appropriate professional
            support, particularly if you may be in immediate
            danger.
            """
        )


    # ========================================================
    # EXPLAINABLE AI
    # ========================================================

    st.subheader(
        "🧠 Why did the AI make this prediction?"
    )

    st.write(
        """
        The machine-learning prediction is based on the
        combination of the PHQ-9 responses and additional
        information supplied to the model.
        """
    )

    explanation_data = pd.DataFrame(
        {
            "Factor": [

                "Little interest or pleasure",

                "Feeling down or hopeless",

                "Sleep difficulty",

                "Low energy",

                "Concentration difficulty",

                "Study pressure",

                "Financial pressure"

            ],

            "Response": [

                q1,

                q2,

                q3,

                q4,

                q7,

                study_pressure,

                financial_pressure

            ]
        }
    )

    st.dataframe(
        explanation_data,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.subheader(
        "🤖 Model Information"
    )

    model_col1, model_col2, model_col3 = st.columns(3)

    with model_col1:

        st.metric(
            "Algorithm",
            "Linear SVM"
        )

    with model_col2:

        st.metric(
            "Kernel",
            "Linear"
        )

    with model_col3:

        st.metric(
            "C",
            "100"
        )


    # ========================================================
    # GENERAL GUIDANCE
    # ========================================================

    st.subheader(
        "💡 General Guidance"
    )

    if prediction == "Minimal":

        st.info(
            """
            The model estimates minimal depressive symptoms.

            Maintaining healthy sleep, physical activity,
            social connection, and study-life balance may
            be helpful.
            """
        )

    elif prediction == "Mild":

        st.info(
            """
            The model estimates mild depressive symptoms.

            Consider monitoring your wellbeing and talking
            with someone you trust if symptoms persist.
            """
        )

    elif prediction == "Moderate":

        st.warning(
            """
            The model estimates moderate depressive symptoms.

            Consider discussing your concerns with a qualified
            mental-health professional.
            """
        )

    elif prediction == "Moderately severe":

        st.warning(
            """
            The model estimates moderately severe symptoms.

            Professional mental-health support is recommended.
            """
        )

    elif prediction == "Severe":

        st.error(
            """
            The model estimates severe depressive symptoms.

            Consider seeking professional mental-health
            support as soon as possible.
            """
        )


    # ========================================================
    # FINAL DISCLAIMER
    # ========================================================

    st.divider()

    st.caption(
        """
        ⚠️ Important: DepressionRiskAI is an educational and
        research prototype. The machine-learning prediction
        is not a medical diagnosis and should not be used to
        make medical decisions.

        Model probability is not equivalent to clinical
        certainty.
        """
    )