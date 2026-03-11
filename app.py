import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("Belbin Role Analysis Dashboard")

# -------------------------------------------------
# Load dataset
# -------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("updated_students_data.csv")

students = load_data()

students_sorted = students.sort_values(by="gpa", ascending=False)

# -------------------------------------------------
# Belbin roles
# -------------------------------------------------

belbin_roles_display = [
"Plant",
"Monitor Evaluator",
"Specialist",
"Implementer",
"Completer Finisher",
"Shaper",
"Coordinator",
"Teamworker",
"Resource Investigator"
]

dataset_roles = [
"plant",
"monitor_evaluator",
"specialist",
"implementer",
"completer_finisher",
"shaper",
"coordinator",
"teamworker",
"resource_investigator"
]

# -------------------------------------------------
# Belbin descriptions
# -------------------------------------------------

belbin_descriptions = {

"Plant": "Creative innovator who generates ideas and solves difficult problems.",
"Monitor Evaluator": "Strategic analytical thinker evaluating ideas objectively.",
"Specialist": "Expert providing deep technical knowledge.",
"Implementer": "Practical organizer turning ideas into actions.",
"Completer Finisher": "Detail oriented person ensuring tasks are completed.",
"Shaper": "Dynamic individual driving the team forward.",
"Coordinator": "Leader clarifying goals and delegating tasks.",
"Teamworker": "Supportive collaborator improving team cohesion.",
"Resource Investigator": "Network builder exploring opportunities."

}

# -------------------------------------------------
# Load embedding model
# -------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

role_embeddings = {
    role: model.encode(desc)
    for role, desc in belbin_descriptions.items()
}

# -------------------------------------------------
# Job Belbin scoring
# -------------------------------------------------

def belbin_score(text):

    job_emb = model.encode(text)

    scores = {}

    for role, emb in role_embeddings.items():

        sim = cosine_similarity([job_emb],[emb])[0][0]

        scores[role] = sim

    total = sum(scores.values())

    scores = {k:v/total for k,v in scores.items()}

    return scores

# -------------------------------------------------
# Improved candidate ranking
# -------------------------------------------------

def best_candidates(job_scores):

    job_vector = np.array([job_scores[r] for r in belbin_roles_display])

    similarities = []

    for _, row in students.iterrows():

        student_vector = row[dataset_roles].values.astype(float)
        student_vector = student_vector / student_vector.sum()

        sim = cosine_similarity([job_vector],[student_vector])[0][0]

        gpa_norm = row["gpa"] / students["gpa"].max()

        capacity = 0.5 * row["X"] + 0.5 * row["Y"]

        final_score = (
            0.6 * sim +
            0.25 * capacity +
            0.15 * gpa_norm
        )

        similarities.append(final_score)

    students["match_score"] = similarities

    return students.sort_values("match_score", ascending=False).head(3)

# -------------------------------------------------
# UPPER SECTION
# -------------------------------------------------

col1, col2 = st.columns(2)

# Left table

with col1:

    st.subheader("Students ranking by GPA")

    st.dataframe(
        students_sorted[["name","surname","gpa"]],
        use_container_width=True
    )

# -------------------------------------------------
# Right section: 9 Box Matrix
# -------------------------------------------------

with col2:

    st.subheader("9-Box Talent Matrix")

    x_low = 1/3
    x_med = 2/3
    y_low = 1/3
    y_med = 2/3

    fig, ax = plt.subplots(figsize=(6,6))

    ax.scatter(students["X"], students["Y"], color="red", s=80)

    for _, row in students.iterrows():

        ax.text(
            row["X"],
            row["Y"],
            row["surname"],
            fontsize=6,
            rotation=45
        )

    ax.axvline(x_low, linestyle="--", color="gray")
    ax.axvline(x_med, linestyle="--", color="gray")
    ax.axhline(y_low, linestyle="--", color="gray")
    ax.axhline(y_med, linestyle="--", color="gray")

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    ax.set_xlabel("Delivery Capacity (X)")
    ax.set_ylabel("Growth & Complexity Capacity (Y)")

    st.pyplot(fig)

# -------------------------------------------------
# LOWER SECTION
# -------------------------------------------------

st.markdown("---")

st.subheader("Insert Job Description")

job_description = st.text_area(
    "Job description",
    height=200
)

# -------------------------------------------------
# Belbin role detection
# -------------------------------------------------

if st.button("Which are the most relevant belbin roles?"):

    scores = belbin_score(job_description)

    st.session_state["job_scores"] = scores

    st.subheader("Belbin Role Coefficients")

    st.json(scores)

    # fig, ax = plt.subplots()

    # ax.bar(scores.keys(), scores.values())

    # plt.xticks(rotation=45)

    # st.pyplot(fig)

# -------------------------------------------------
# Best candidates
# -------------------------------------------------

if "job_scores" in st.session_state:

    if st.button(
        "Which are the best candidates?",
        help="""
Final Candidate Score =

0.6 × Belbin Similarity
+ 0.25 × Capacity Score
+ 0.15 × GPA

Where:

Belbin Similarity = cosine similarity between job Belbin profile
and candidate Belbin profile.

Capacity Score = 0.5 × X + 0.5 × Y

X = Delivery Capacity
Y = Growth & Complexity Capacity

GPA normalized relative to best student.

Top 3 candidates with highest final score are selected.
"""
    ):

        top3 = best_candidates(st.session_state["job_scores"])

        st.session_state["top3"] = top3


# -------------------------------------------------
# Candidate toggle visualization
# -------------------------------------------------

if "top3" in st.session_state:

    top3 = st.session_state["top3"]

    st.subheader("Top 3 Candidates")

    st.dataframe(
        top3[["name","surname","gpa","match_score"]],
        use_container_width=True
    )

    # Candidate selector
    candidate_labels = [
        f"{row['name']} {row['surname']}"
        for _, row in top3.iterrows()
    ]

    selected_candidate = st.radio(
        "Select candidate to compare with the job profile:",
        candidate_labels
    )

    selected_row = top3[
        (top3["name"] + " " + top3["surname"]) == selected_candidate
    ].iloc[0]

    # Prepare vectors
    job_vector = np.array([
        st.session_state["job_scores"][r]
        for r in belbin_roles_display
    ])

    student_vector = selected_row[dataset_roles].values.astype(float)
    student_vector = student_vector / student_vector.sum()

    # Plot comparison
    x = np.arange(len(belbin_roles_display))

    fig, ax = plt.subplots()

    ax.plot(
        x,
        job_vector,
        marker="o",
        linewidth=3,
        label="Job Profile"
    )

    ax.plot(
        x,
        student_vector,
        marker="o",
        linewidth=3,
        label=selected_candidate
    )

    ax.set_xticks(x)
    ax.set_xticklabels(belbin_roles_display, rotation=45)

    ax.set_ylabel("Normalized Score")

    ax.legend()

    st.pyplot(fig)