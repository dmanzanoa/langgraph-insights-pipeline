# LangGraph Insights Pipeline

This repository contains a modular Python implementation of the *LangGraph insights* workflow. The goal of the pipeline is to ingest raw chat data from Amazon S3, compress each conversation into a structured summary using an Anthropic model running on AWS Bedrock, compute aggregated metrics and language statistics, and finally generate higher‑level business insights. The system also produces monthly trend reports (tendencias) and project‑specific insights.


## Architecture Overview

At a high level the pipeline performs the following steps for each data source (`subsidio`, `no_subsidio` and `recomendador`):

1. **Load data** – Download all parquet files from the configured S3 prefix.
2. **Preprocess** – Clean and normalise the data, then group messages into per‑client conversations.
3. **Summarise** – Use the Bedrock *compress* model to reduce each conversation to a structured summary. A small thread pool speeds up this step.
4. **Generate insights** – Build a structured dataframe from the summaries, compute aggregated metrics and dominant terms, and feed these into the main LLM along with a schema prompt (defined in `prompts.py`). The LLM returns a JSON object capturing a variety of business insights.
5. **Validate and retry** – Parse the returned JSON and check that required keys are present. If keys are missing the pipeline constructs a stricter prompt highlighting the omissions and retries up to `MAX_RETRIES` times before giving up.
6. **Persist** – Valid insights are written back to S3. Fatal errors are logged and persisted separately.

After the main insights are saved the pipeline continues with:

7. **Monthly tendencias** – For each month present in the dataset the code aggregates metrics, extracts dominant language, and calls the LLM to generate a monthly trend report. The resulting JSON is validated and persisted.
8. **Per‑project insights** – Finally, the pipeline iterates over each project (`subProjectInfo`) and generates project‑specific insights using the same aggregation logic.

The flow is orchestrated using [LangGraph](https://github.com/langchain-ai/langgraph), which allows nodes to be connected declaratively and retried based on success or failure. Each node operates on a mutable state dictionary and returns a new state; routing logic determines whether to retry, continue or bail out.

## Module Breakdown

| Module | Purpose |
|-------|---------|
| `config.py` | Central configuration: reads AWS region, S3 buckets and Bedrock model identifiers from environment variables; exposes boto3 clients and a `DATA_SOURCES` mapping. |
| `state.py` | Defines the `PipelineState` type (a simple `dict`) and a helper for constructing fatal error objects with timestamps. |
| `prompts.py` | Contains all the multi‑line schema prompts used when prompting the LLMs. Keeping prompts in a dedicated module makes them easy to maintain. |
| `preprocessing.py` | Text pre‑processing helpers: normalise project names, merge messages into conversational turns and full conversations, and filter recent data. |
| `aggregation.py` | Functions to compute aggregated metrics (counts, percentages and top‑k values) and to extract dominant terms using TF‑IDF. |
| `data_loading.py` | Utility to fetch all parquet files under an S3 prefix and concatenate them into a single `pandas` DataFrame. |
| `summarization.py` | Wraps calls to the Bedrock compress model, validates the compressed output against a schema and retries if necessary. |
| `insights.py` | Wraps calls to the main Bedrock model, parses and validates the returned JSON, builds structured DataFrames from compressed summaries and constructs retry prompts. |
| `pipeline_nodes.py` | Implements individual pipeline steps (nodes) that operate on the shared state. These include loading data, preprocessing, summarisation, generating insights, validating outputs, producing monthly tendencias, generating project‑level insights and persisting results to S3. |
| `graph_builder.py` | Assembles the nodes into three LangGraph pipelines (insights, tendencias and subproject) and wires up the conditional routing. |
| `app.py` | Provides a simple entry point that runs the compiled pipelines for all configured data sources. Invoke `python -m app` to execute the workflow. |

## Running the Pipeline

Before running the pipeline you must configure the environment with the appropriate AWS credentials and override any defaults as needed:

```bash
export AWS_REGION=us-east-1
export S3_BUCKET=my-input-bucket
export OUTPUT_BUCKET=my-output-bucket
export MODEL_MAIN=my-bedrock-model
export MODEL_COMPRESS=my-bedrock-compress-model
# Optional: customise the data prefixes
export DATA_SOURCE_SUBSIDIO="path/to/subsidio/"
export DATA_SOURCE_NO_SUBSIDIO="path/to/no_subsidio/"
export DATA_SOURCE_RECOMENDADOR="path/to/recomendador/"
```

Install the required dependencies (for example via pip):

```bash
pip install boto3 pandas langgraph json-repair numpy scikit-learn
```

Then run the pipeline:

```bash
python -m app
```

During execution the script will log progress to the console. Results are written back to the output S3 bucket under the prefixes `comercial/insights_graph_test/`, `comercial/insights_tendencias_test/` and `comercial/insights_by_project_test/`.

## Development Notes

* The pipeline deliberately avoids using Jupyter and instead exposes a CLI entry point to aid reproducibility and version control. Notebooks can still be used for experimentation but the production code lives in Python modules.
* Long prompts containing JSON schemas are stored as triple‑quoted strings in `prompts.py`. Avoid inserting business logic into prompt definitions; instead compute metrics in Python and pass them as structured input.
* The code uses small thread pools to parallelise summarisation. If you are processing a very large number of conversations you may wish to increase the `max_workers` values or migrate to an asynchronous architecture.
* All external calls to Bedrock are wrapped in helper functions so they can be mocked during testing. The `MAX_RETRIES` constant in `config.py` controls how many times the pipeline will retry when encountering invalid JSON.

## License

This project is provided as is under the MIT License.
