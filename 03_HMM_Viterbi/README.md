# PathSnap

PathSnap matches a raw GPS track to a directed road graph. The local Route Lab lets you select any supplied student-test trip, run live inference, explore GPS and predicted routes, inspect individual observations, and export the decoded route as GeoJSON. The Experiments view shows measured model and decoder comparisons.

## Start on this laptop

From this directory, run `./Start-PathSnap.ps1` in PowerShell and open `http://127.0.0.1:8765`. The script uses `.venv` here or the existing `.venv` in the parent workspace. The dataset is already imported on this laptop. FastAPI docs are at `/docs`. The Route Lab itself uses local road geometry and works without online map tiles.

For a fresh checkout on Windows with Python 3.11:

```powershell
py -3.11 -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install -r requirements.txt
& $python scripts/import_dataset.py 'C:\path\to\Student folder.zip'
./Start-PathSnap.ps1
```

On Linux, create and activate a Python 3.11 virtual environment, install `requirements.txt`, import the ZIP, and run `python -m uvicorn pathsnap.api:app --host 127.0.0.1 --port 8765`.

## Run the experiments from raw data

```powershell
& $python -m pip install -r requirements-dev.txt
& $python -m pathsnap.preprocessing.cli inventory
& $python -m pathsnap.preprocessing.cli process --config configs/default.json
& $python -m pathsnap.preprocessing.cli validate
& $python -m pathsnap.experiments --trials 12
& $python -m pathsnap.route_experiments
& $python scripts/build_visualizations.py
& $python -m pytest -q
```

Use the `& $python` defined above or the Python executable in your active environment. The run reads 80 training trips and compares models on ten student-test trips. Of the 80 training trips, 79 produce geometrically supported point labels. Three grouped CV folds keep each trip together. The classifier uses 13 features. Initial model selection uses CV observation Top-1, with macro F1 and log loss as tie-breakers. Optuna TPE tunes the selected model using CV macro F1. The decoder grid uses student-test mean ordered route F1 as its primary criterion. Both steps log all trials and settings.

`models/` contains the saved model files, feature order, preprocessing defaults and decoder configuration. Use the pinned runtime for the bundled joblib artifacts. Rebuild them with the training module when changing library versions.

`results/` contains CSV and JSON evidence for the data audit, CV folds, each model and Bayesian trial, decoder settings, per-trip comparison and live API run. `scripts/build_visualizations.py` generates plots and an interactive Folium map from those runs. The development Folium map includes its supplied student-test reference for comparison. No reference geometry is returned by the Route Lab API.

Reference agreement is measured on a subset of GPS observations that can be aligned using geometry, heading and sequence constraints. Ordered sequence F1 compares against the supplied chronological arc list. Continuity measures linked adjacent observations among the pairs eligible for a transition. The Experiments view shows development measurements. Reserved results are generated separately by the live evaluation endpoint.

## Reserved evaluation and LAN mode

Keep organizer data outside version control. For the supplied evaluator ZIP, import it into the ignored `data/evaluator/` directory and start the launcher. The importer rejects trip IDs found in either development partition and keeps the reference routes in a separate directory.

```powershell
& $python scripts/import_dataset.py 'C:\path\to\evaluator_test.zip' --evaluator
./Start-PathSnap.ps1
& $python scripts/verify_api.py --partition evaluator_test --count 10 --require-reference --out data/evaluator/api_verification.json
```

Run verification in a second terminal while the server is listening. For a custom layout, put each reserved input in `<evaluator-root>/<trip-id>/` with `<trip-id>.track`, `.nodes`, and `.arcs`. Put reference routes in the private reference directory as `<trip-id>.route` or `<trip-id>/<trip-id>.route`. If references are co-located with their inputs, omit `PATHSNAP_EVALUATOR_REFERENCE_ROOT`. The inference loader reads only the three input files. The API reads the reference after predictions are complete and returns only aggregate evaluation values.

```powershell
$env:PATHSNAP_EVALUATOR_ROOT = 'D:\private\evaluator-trips'
$env:PATHSNAP_EVALUATOR_REFERENCE_ROOT = 'D:\private\reference-routes'
./Start-PathSnap.ps1 -Lan
```

On laptop 2, open `http://<laptop-1-LAN-IP>:8765` in a browser on the same network. The browser runs the UI and calls laptop 1 over HTTP. When reserved inputs are configured, the trip selector opens the evaluator partition first. Keep the server console visible for request logs.

## API and evidence

| Endpoint | Behavior |
| --- | --- |
| `GET /api/health` | Loaded pipeline and readiness |
| `GET /api/trips` | Available partition and trip metadata |
| `POST /api/match` | Fresh inference for `{"trip_id":"00000000","partition":"student_test"}` |
| `GET /api/experiments` | Measured development experiment summaries |

```powershell
& $python scripts/verify_api.py --compare results/route_comparison.json --out results/api_verification.json
& $python scripts/capture_evidence.py --folium results/maps/student_trip_00000007.html
& $python scripts/build_report.py
```

The report builder uses the browser screenshots, `results/receiver_excerpt.txt` from the API receiver logs and the experiment output files. Playwright uses the installed Chrome browser by default. The Folium map loads Leaflet assets from jsDelivr. The assignment calls for a public GitHub repository and one consolidated PDF. This workspace contains the local Git repository and report for review.

To include verified reserved results in the local report, pass `--evaluator data/evaluator/api_verification.json` to the report builder.

AI assistance from OpenAI Codex is disclosed in the report. Source, configurations and measured outputs are retained so the team can reproduce the experiments and explain the implementation.
