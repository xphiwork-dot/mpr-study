# Leakage check (all 12 users)

Output of `scripts/leak_check.py --show-example`, re-run on 2026-11-01 for this commit. A question leaks if it shares a number with its own gold statements (regex `\d+(\.\d+)?`). The 3+ digit row ignores short numbers that match by chance. 32,399 questions = 32,400 minus one null question.

```
users [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]: 32399 questions
hop                 2      3      4      5      6      7      8      9     10    all
any number       0.13   0.07   0.27   0.60   0.65   0.71   0.75   0.80   0.82   0.53
3+ digit nums    0.10   0.05   0.20   0.42   0.44   0.50   0.53   0.60   0.62   0.38

example (user 0, RC000000#07), shared numbers ['13', '2024']:
The departure point of the Celestial Quest Expedition is the same as that of another expedition, sharing a common starting point. This other expedition concludes on July 13, 2024. Calculating backwards, 244 days prior to this conclusion marks another important date. Adding 33 days to this earlier date arrives at a new significant moment in the sequence of events. Furthermore, extending this timeline by an additional 13 days takes us to another key end date. What is the end date obtained by adding 13 days to the date 33 days after 244 days before July 13, 2024?
```

Command: `.venv/bin/python scripts/leak_check.py --show-example`
