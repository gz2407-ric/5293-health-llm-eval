# Model and Data Card

## Project

**Evaluating LLM Architectures for Personalized Lifestyle Health Recommendations**

This project compares LLM system architectures for generating non-clinical lifestyle, diet, and exercise recommendations. It is designed as a course project evaluation study, not as a medical product.

## Intended Use

The system is intended for:

- Course-project demonstration
- Evaluation of GenAI architecture choices
- Non-clinical lifestyle recommendation research
- Comparison of prompting, RAG, and multi-step pipeline designs

## Not Intended Use

The system is not intended for:

- Medical diagnosis
- Medication, dosage, or treatment advice
- Emergency health advice
- Clinical decision-making
- Replacement of healthcare professionals
- Real patient deployment

## Data

### Scenario Dataset

The evaluation uses **50 synthetic health-profile scenarios**. These are manually curated / LLM-assisted synthetic user profiles and do not contain real patient data.

Each scenario includes:

- scenario_id
- scenario_type
- age
- BMI
- lifestyle_habits
- health_goal
- risk_condition
- user_question
- expected_safety_warning
- expected_guideline_topic

Scenario categories:

| Scenario Type | Count |
|---|---:|
| healthy_adult | 10 |
| overweight_obesity | 10 |
| hypertension | 10 |
| prediabetes_diabetes_risk | 10 |
| injury_limited_mobility | 5 |
| pregnancy_special_caution | 5 |

### Guideline Corpus

The RAG corpus contains **50 guideline chunks** from public CDC/WHO-style health guidance. Each chunk includes:

- chunk_id
- source
- source_title
- topic
- retrieval_keywords
- text
- url

The guideline corpus covers:

- Physical activity
- Healthy diet
- Weight management
- Diabetes prevention
- Hypertension lifestyle
- Sodium reduction
- Injury-safe activity
- Pregnancy physical activity
- Sleep habits
- Sedentary behavior

## Systems Compared

| System | RAG | Multi-step | Safety Classifier | Description |
|---|---:|---:|---:|---|
| Baseline | No | No | No | Single-prompt generation |
| RAG only | Yes | No | No | Retrieves guideline chunks before generation |
| Multi-step only | No | Yes | Yes | Parses profile, flags risk, generates and checks safety |
| Full pipeline | Yes | Yes | Yes | Combines retrieval, structured generation, and safety checking |

## Evaluation

The systems are evaluated on:

- Relevance
- Personalization
- Guideline Alignment
- Safety

Evaluation methods include:

- LLM-as-judge scoring
- Paired statistical testing
- Retrieval Recall@k
- Judge self-consistency check
- Blind human validation
- Scenario-type breakdown
- Output length analysis
- Counterfactual personalization case study
- Qualitative error taxonomy

## Ethical Considerations

- The project uses synthetic scenarios, not real patient records.
- The system explicitly avoids diagnosis, medication, and treatment recommendations.
- High-risk users should be advised to consult healthcare professionals.
- Outputs should be interpreted as general wellness suggestions only.
- The project acknowledges limitations of LLM-as-judge evaluation and validates a subset with blind human ratings.

## Known Limitations

| Limitation | Why it matters |
|---|---|
| Synthetic scenarios | May not fully represent real user behavior. |
| Limited guideline corpus | Only covers selected CDC/WHO-style lifestyle guidance. |
| English-only evaluation | Does not test multilingual generalization. |
| LLM-as-judge bias | Scores may reflect judge preferences or verbosity bias. |
| Limited human validation size | Human validation covers a subset, not all outputs. |
| Non-clinical scope | Results do not apply to clinical medical advice. |

## Recommended Use in Demo

The safest demo mode is to use precomputed outputs and scores. A live custom-profile mode can be provided as a secondary feature, but it should clearly state that it calls the live API and may take time.
