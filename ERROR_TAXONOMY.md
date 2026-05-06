# Error Taxonomy and Qualitative Case Study Notes

This file summarizes qualitative patterns observed from the generated recommendations and the automatically selected case-study candidates. It is meant to support the final report's error analysis section.

## Error / Phenomenon Taxonomy

| Category | Meaning | Where it appeared | How to discuss it |
|---|---|---|---|
| Generic advice | The output is safe and broadly relevant, but could apply to many users. | Mainly affects personalization. | Do not overclaim personalization gains from average scores alone. |
| Unsupported guideline-like claims | The answer gives plausible public-health advice but does not clearly ground it in retrieved evidence. | Baseline and Multi-step-only systems. | This explains why non-RAG systems score lower in guideline alignment. |
| Missing or weak caution | The output is generally safe but does not give enough caution for high-risk users. | Some baseline and RAG-only outputs. | This motivates the risk-flagging and safety-classifier components. |
| Over-conservatism | The output gives repeated caution/disclaimer language and may become less practical. | More common in Multi-step-only outputs. | This helps explain why Multi-step-only can score high on safety but not dominate other metrics. |
| Specificity-safety trade-off | More specific guideline-grounded advice can appear slightly less conservative than a safety-first response. | Full pipeline vs Multi-step-only. | This is a balanced finding, not a failure. |
| Limited personalization separation | All systems used structured profiles reasonably well, so average personalization scores were close. | Aggregate personalization results. | Use counterfactual analysis as additional evidence. |

## Recommended Case Study Types

Use `results/case_studies_notes.csv` to pick concise examples for the report:

1. **RAG improves guideline alignment over baseline**  
   Show how retrieved CDC/WHO chunks make the RAG response more grounded.

2. **Multi-step improves safety over baseline**  
   Show risk flags or safety revision leading to a more cautious answer.

3. **Full pipeline strong guideline grounding**  
   Show the complete system using retrieval and risk-aware generation together.

4. **Specificity-safety trade-off**  
   Show a case where Full pipeline is more specific/grounded while Multi-step-only is more conservative.

5. **Small personalization difference**  
   Show that standard prompting already captures basic profile fields, explaining the modest average gap.

## Suggested Report Wording

The qualitative analysis suggests that RAG mainly reduces unsupported guideline-like claims by grounding outputs in retrieved CDC/WHO snippets. Multi-step prompting improves risk awareness, but may also make answers more conservative. The full pipeline provides the strongest grounding, while safety gains are more mixed, suggesting a trade-off between specificity and conservative safety behavior.
