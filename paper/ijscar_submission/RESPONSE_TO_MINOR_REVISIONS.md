# Response to expedited review minor revisions

Dear Prof. Hwang and the IJSCAR Editorial Team,

Thank you for the expedited review and for the encouraging assessment. I have completed each requested minor revision:

1. **Nested-policy warning timing.** Table 5 now includes the nested selected policy: 54 of 90 engines are warned at or before the final-window start, 34 receive a later warning, two receive no warning, and the median lead is one cycle among warned engines. These values are recorded in `benchmark_results/maintenance_warning/model_selected_observable_20260814_robust/results.json`.
2. **Repository links.** Reference [3] now prints the tiny-N-CMAPSS repository URL and a direct link to pinned commit `b915997ff8d571f4e9d091954d6556835f212ded`. I verified that both the tiny-N-CMAPSS commit and the project repository resolve publicly. The project repository now includes the corrected evaluation scripts, outer-test predictions, thresholds, fold metrics, first-warning records, result summaries, manuscript source, and revised PDF.
3. **Abstract.** The abstract now retains only the primary matched-budget comparison: recall increases from 55.4% to 79.3%, a 23.9-percentage-point gain with a 95% complete-engine bootstrap interval from 18.3 to 29.6 points. The nested-selection and telemetry-only timing results remain in the Results section.
4. **Formatting.** Table 3 now uses the same LaTeX caption structure as the other tables. The redundant embedded dashboard headings and summary footers were removed from all three figures so their presentation is consistent and the manuscript captions carry the explanatory text.

I also clarified the seed-robustness reporting. The manuscript now states that
the five-seed mean recall gain is 22.6 percentage points (range 21.9 to 23.9),
while the 18.3-to-29.6-point interval reflects complete-engine resampling for
the primary run (seed 2026) rather than model-seed variability. The Conclusion
now makes clear that first-warning timing was evaluated for the primary run
only.

Thank you again for the opportunity to revise the paper.

Best regards,

Prahalad Talur
Eastlake High School
ptalur09@gmail.com
