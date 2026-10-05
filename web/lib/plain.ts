// Plain-language text for the user interface.
// Writing rules (based on ASD-STE100 Simplified Technical English), see docs/12_writing_style.md:
//  - One idea in each sentence. Procedures: max 20 words. Descriptions: max 25 words.
//  - Use the active voice and the imperative ("Select a station.").
//  - Use one word for one meaning. Use the terms in GLOSSARY only.
//  - Use simple words: "use" (not "utilize"), "see" (not "visualize"), "chance" (not "probability").

export const TERMS = {
  extremeRain: "Extreme rain day",
  hotDay: "Hot day",
  dryPeriod: "Dry period",
  suddenChange: "Sudden dry-to-wet change",
  chance: "Chance",
  confidence: "Confidence",
  uncertainty: "Uncertainty",
} as const;

export const GLOSSARY: { term: string; text: string }[] = [
  { term: TERMS.extremeRain, text: "A day with more rain than 95 of 100 rainy days at that station." },
  { term: TERMS.hotDay, text: "A day that is hotter than 90 of 100 days at the same time of year." },
  { term: TERMS.dryPeriod, text: "15 or more days in a row with almost no rain." },
  { term: TERMS.suddenChange, text: "A very dry month changes to a very wet month in 30 days or less." },
  { term: TERMS.chance, text: "How likely the event is, from 0% (no chance) to 100% (certain)." },
  { term: TERMS.confidence, text: "How clear the answer of the model is. A high value is a clear answer." },
  { term: TERMS.uncertainty, text: "How much the model versions disagree. High uncertainty means: check other sources." },
];

export const NAV: { section: string; links: [href: string, label: string][] }[] = [
  { section: "", links: [["/", "Home"]] },
  { section: "Explore", links: [["/map", "Map"], ["/climate", "Rain and temperature"], ["/extremes", "Extreme events"]] },
  { section: "Forecast", links: [["/run", "Get a forecast"]] },
  { section: "Research", links: [["/compare", "Compare models"], ["/experiments", "Experiments"], ["/dataset", "Data quality"]] },
  { section: "", links: [["/docs", "Help"], ["/admin", "Admin"]] },
];
