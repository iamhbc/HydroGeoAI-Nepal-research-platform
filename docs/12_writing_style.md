# 12. Writing style for the user interface

The public pages use rules from **ASD-STE100 Simplified Technical English**. The goal: a user reads each sentence
once and understands it, including users with English as a second language.

## Rules
1. Write one idea in each sentence.
2. Keep instructions to 20 words or fewer. Keep descriptions to 25 words or fewer.
3. Use the active voice: "We remove bad values." Not: "Bad values are removed."
4. Use the imperative for instructions: "Select a station." "Click Run analysis."
5. Use one word for one meaning. Use only the terms in `web/lib/plain.ts` (`TERMS`, `GLOSSARY`).
6. Use simple words: *use* (not utilise), *see* (not visualise), *chance* (not probability), *check* (not validate).
7. Write numbers with units: "15 or more days", "0% to 100%".
8. Put warnings before the action they apply to. Start with the result: "Test data. Do not use it to make decisions."

## Approved terms
| Use | Do not use |
|---|---|
| Extreme rain day | extreme_wet_day, R95p event |
| Hot day | TX90p exceedance |
| Dry period | CDD, dry spell ≥ 15 d |
| Sudden dry-to-wet change | hydroclimatic whiplash (research pages only) |
| Chance | probability, likelihood |
| Confidence | 1 − normalised entropy |
| Uncertainty | epistemic uncertainty, mutual information |

## Where technical words are allowed
The research pages (Compare models, Experiments, Data quality, Help and methods, Admin) are for researchers.
They can use technical terms, but each page starts with one plain sentence that says what the page is for.

## Design rules that support the text
* One main action on each page. It is the first button.
* Touch targets are 44 px or more. Keyboard focus is always visible.
* Colours pass WCAG AA contrast in light and dark mode.
* The home page loads from one cached request (`GET /overview`) and is rendered on the server.
