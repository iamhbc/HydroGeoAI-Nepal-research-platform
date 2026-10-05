# 10. Hugging Face release

Three artifacts, generated from the actual data and experiments so that the documentation matches them.

| Artifact | Repo name | Contents | Card |
|---|---|---|---|
| Dataset | `hydrogeoai-nepal-dataset` | observations (QC flags, events, SPI), stations + static features, thresholds, QC flags, homogeneity, trends, ETCCDI, whiplash, geometries, DEM, metadata, provenance | provenance, variables, coverage, missingness, preprocessing, licensing, limitations, ethics (`hf/cards.py::dataset_card`) |
| Model | `hydrogeoai-nepal-model` | `config.json` (normalisation, channels, threshold, temperature, conformal quantiles, versions), `member_*.pt` ensemble, `encoder/` (HF layout) | architecture, training data, intended use, limitations, evaluation with CIs, uncertainty, hypotheses, failure cases (`model_card`) |
| Space | `hydrogeoai-nepal-demo` | Gradio app (`hf/space/app.py`), model, demo data slice, bundled package | Input → Model → Prediction → Explanation → Uncertainty, with versions |

```bash
make hf-export        # stage under hf/export/ (local only)
make hf-publish-dry   # dry run: shows repo ids, file counts and sizes, and the account you're logged in as
make hf-login         # `hf auth login`: paste a *write* token from https://huggingface.co/settings/tokens
make hf-publish       # creates + uploads the three repos under your username
```
Equivalent CLI: `hydrogeoai hf whoami | export | publish [--namespace org] [--push] [--private]`.

Verified locally:
* the staged dataset loads with `datasets.load_dataset("hf/export/hydrogeoai-nepal-dataset", "observations")`
  through the card's `configs:` block (`stations` config too);
* the staged Space runs standalone from its folder (`python hf/export/hydrogeoai-nepal-demo/app.py`), using the
  bundled package, model and demo data, exactly as it would on Hugging Face (Gradio SDK).

The model card is written into each experiment's model directory, and the dataset card is generated at export
time. Check licences before publishing real station data; use `--private` for a first upload if you want to
review the pages before making them public.
