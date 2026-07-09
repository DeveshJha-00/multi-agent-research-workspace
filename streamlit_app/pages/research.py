"""Browser UI for durable multi-agent research and repository analysis."""

from uuid import uuid4

import streamlit as st

from streamlit_app.utils.api_client import (
    cancel_research_job,
    create_research_job,
    dataset_upload,
    document_upload_rag,
    download_artifact,
    get_datasets,
    get_repositories,
    get_research_job,
    get_research_jobs,
    get_research_result,
    repository_upload,
    retry_research_job,
    stream_research_events,
)

st.set_page_config(page_title="Agentic Research", page_icon="🧭", layout="wide")

if "session_id" not in st.session_state:
    st.warning("Open a workspace first.")
    st.page_link("home.py", label="Go to home")
    st.stop()

st.session_state.setdefault("research_runs", [])
st.session_state.setdefault("uploaded_files", {})
st.session_state.setdefault("uploaded_datasets", {})
st.session_state.setdefault("uploaded_repositories", {})


def _render_event(event: dict) -> None:
    st.write(f"**{event['stage'].replace('_', ' ').title()}** - {event['message']}")
    details = event.get("details", {})
    if event["event"] == "repository_inventory_completed":
        languages = ", ".join(
            f"{name}: {count}" for name, count in details.get("languages", {}).items()
        )
        st.caption(
            f"Files: {details.get('file_count', 0)}"
            + (f" | Languages: {languages}" if languages else "")
        )
    elif event["event"] == "repository_inspection_completed":
        technologies = ", ".join(details.get("technologies", [])) or "none detected"
        st.caption(
            f"Inspected {details.get('files_inspected', 0)} manifests/entry points | "
            f"Technology markers: {technologies}"
        )
    elif event["event"] == "repository_search_completed":
        terms = ", ".join(details.get("terms", [])) or "none"
        st.caption(f"Matches: {details.get('matches', 0)} | Search terms: {terms}")
    elif event["event"] == "repository_synthesis_completed":
        st.caption(f"Generated {details.get('characters', 0)} characters of grounded explanation")


def watch_job(task_id: str) -> None:
    """Render persisted events live, then load the completed durable result."""
    last_sequence = 0
    with st.status(f"Watching research job {task_id[:8]}...", expanded=True) as activity:
        for event in stream_research_events(task_id, st.session_state.session_id, last_sequence):
            if event.get("error"):
                st.warning(event["error"])
                break
            last_sequence = event["sequence"]
            _render_event(event)
            activity.update(label=f"{event['progress']}% | {event['message']}")

        job = get_research_job(task_id, st.session_state.session_id)
        if job.get("status") == "completed":
            result = get_research_result(task_id, st.session_state.session_id)
            if not result.get("error"):
                known = {item["task_id"] for item in st.session_state.research_runs}
                if task_id not in known:
                    st.session_state.research_runs.insert(0, result)
                activity.update(label="Research complete", state="complete")
                return
        if job.get("status") == "cancelled":
            activity.update(label="Research cancelled", state="error")
        elif job.get("status") == "failed":
            activity.update(label="Research failed", state="error")
            st.error(job.get("error", "The job failed."))
        elif job.get("error"):
            activity.update(label="Unable to read job status", state="error")
        else:
            activity.update(label=f"Connection closed | job is {job.get('status')}")


def start_job(objective: str, available_data: list[str]) -> None:
    created = create_research_job(
        objective.strip(),
        st.session_state.session_id,
        available_data,
        idempotency_key=str(uuid4()),
    )
    if created.get("error"):
        st.error(created["error"])
    else:
        watch_job(created["task_id"])


st.title("Multi-agent research")
st.caption(
    f"Workspace: {st.session_state.session_id} | jobs survive refreshes and service restarts"
)
if st.button("Back to chat"):
    st.switch_page("pages/chat.py")

with st.sidebar:
    st.header("Workspace inputs")
    input_type = st.radio("Upload type", ["Document", "Dataset", "Repository"])

    if input_type == "Document":
        uploaded = st.file_uploader("PDF or TXT", type=["pdf", "txt"], key="research_document")
        description = st.text_input(
            "Document description (optional)",
            key="research_document_description",
        )
        if st.button("Index document", disabled=not uploaded):
            with st.spinner("Indexing document in Qdrant..."):
                result = document_upload_rag(uploaded, description, st.session_state.session_id)
            if not result.get("error"):
                st.session_state.uploaded_files[result["document_id"]] = result
                st.success(f"Indexed {result['filename']}")
            else:
                st.error(result.get("error", "Document upload failed."))
    elif input_type == "Dataset":
        uploaded = st.file_uploader(
            "CSV, JSON, or Excel", type=["csv", "json", "xlsx"], key="research_dataset"
        )
        description = st.text_input("Dataset description", key="research_dataset_description")
        if st.button("Store dataset", disabled=not uploaded):
            with st.spinner("Parsing and storing dataset..."):
                result = dataset_upload(uploaded, description, st.session_state.session_id)
            if not result.get("error"):
                st.session_state.uploaded_datasets[result["dataset_id"]] = result
                st.success(f"Stored {result['filename']} ({result['row_count']} rows)")
            else:
                st.error(result.get("error", "Dataset upload failed."))
    else:
        uploaded = st.file_uploader("Source repository ZIP", type=["zip"], key="repository_zip")
        description = st.text_input("Repository description", key="repository_description")
        if st.button("Store repository", disabled=not uploaded):
            with st.spinner("Validating and storing safe source files..."):
                result = repository_upload(uploaded, description, st.session_state.session_id)
            if not result.get("error"):
                st.session_state.uploaded_repositories[result["repository_id"]] = result
                action = "Reused" if result.get("reused") else "Stored"
                st.success(f"{action} {result['filename']} ({result['file_count']} source files)")
            else:
                st.error(result.get("error", "Repository upload failed."))
        st.caption(
            "ZIPs are inspected as untrusted text. Vendor folders, binaries, symlinks, and unsafe "
            "paths are ignored; uploaded code is never executed."
        )

    st.divider()
    st.subheader("Available data")
    for item in st.session_state.uploaded_files.values():
        st.caption(f"Document | {item['filename']}")
    datasets = get_datasets(st.session_state.session_id)
    for item in datasets:
        st.caption(f"Dataset | {item['filename']} | {item['row_count']} rows")
    repositories = get_repositories(st.session_state.session_id)
    for item in repositories:
        languages = ", ".join(item.get("languages", {})) or "unknown language"
        st.caption(f"Repository | {item['filename']} | {item['file_count']} files | {languages}")

    if st.button("New workspace"):
        for key in list(st.session_state):
            del st.session_state[key]
        st.switch_page("home.py")

available_data = [
    f"Uploaded document: {item['filename']}" for item in st.session_state.uploaded_files.values()
]
available_data.extend(
    f"Dataset ID {item['dataset_id']}: {item['filename']} - {item.get('description', '')}"
    for item in datasets
)
available_data.extend(
    f"Repository ID {item['repository_id']}: {item['filename']} - {item.get('description', '')}"
    for item in repositories
)

research_tab, repository_tab = st.tabs(["General research", "Repository analysis"])

with research_tab:
    objective = st.text_area(
        "Research objective",
        height=150,
        placeholder=(
            "Example: Analyze the sales dataset, compare regional performance, create a chart, "
            "and produce an evidence-backed report."
        ),
    )
    if st.button("Start specialist team", type="primary", disabled=len(objective.strip()) < 10):
        start_job(objective, available_data)

with repository_tab:
    if not repositories:
        st.info("Upload a repository ZIP from the sidebar to enable repository analysis.")
    else:
        selected_repository = st.selectbox(
            "Repository",
            repositories,
            format_func=lambda item: f"{item['filename']} ({item['file_count']} files)",
        )
        focus = st.selectbox(
            "Analysis focus",
            [
                "Architecture and entry points",
                "Dependencies and technology stack",
                "Tests and code-quality signals",
                "Security-sensitive code paths",
                "Custom objective",
            ],
        )
        default_objective = (
            f"Analyze repository {selected_repository['repository_id']} with focus on "
            f"{focus.lower()}. Cite relevant file paths and line matches, identify limitations, "
            "and produce an evidence-backed report."
        )
        repository_objective = st.text_area(
            "Repository-analysis objective",
            value=default_objective,
            height=150,
            key=f"repository-objective-{selected_repository['repository_id']}-{focus}",
        )
        st.caption(
            "The repository specialist checkpoints inventory, inspection, search, and evidence "
            "stages. Its activity remains reconnectable from durable job history."
        )
        if st.button(
            "Analyze repository",
            type="primary",
            disabled=len(repository_objective.strip()) < 10,
        ):
            start_job(repository_objective, available_data)

jobs = get_research_jobs(st.session_state.session_id)
if jobs:
    with st.expander("Durable job history", expanded=False):
        for job in jobs:
            columns = st.columns([5, 2, 1, 1])
            columns[0].write(f"{job['objective'][:90]} | `{job['task_id'][:8]}`")
            columns[1].write(f"{job['status']} | {job['progress']}%")
            if columns[2].button("Watch", key=f"watch-{job['task_id']}"):
                watch_job(job["task_id"])
            if job["status"] in {"queued", "running", "cancel_requested"}:
                if columns[3].button("Cancel", key=f"cancel-{job['task_id']}"):
                    cancel_research_job(job["task_id"], st.session_state.session_id)
                    st.rerun()
            elif job["status"] in {"failed", "cancelled"}:
                if columns[3].button("Retry", key=f"retry-{job['task_id']}"):
                    retry_research_job(job["task_id"], st.session_state.session_id)
                    st.rerun()

for run_index, run in enumerate(st.session_state.research_runs):
    st.divider()
    st.subheader(f"Research result {len(st.session_state.research_runs) - run_index}")
    st.markdown(run.get("content", "No report returned."))

    critique = run.get("critique", {})
    with st.expander("Evidence audit"):
        st.metric("Coverage score", f"{critique.get('coverage_score', 0):.0%}")
        st.write("Approved" if critique.get("approved") else "Completed with limitations")
        for problem in critique.get("problems", []):
            st.warning(problem)

    with st.expander("Specialist activity"):
        for worker in run.get("worker_results", []):
            st.markdown(f"**{worker['agent']}** - {worker['instruction']}")
            st.write(worker.get("summary", ""))
            st.caption(
                f"Tool calls: {worker.get('tool_calls', 0)} | "
                f"Evidence items: {len(worker.get('evidence_ids', []))}"
            )
            if worker.get("error"):
                st.error(worker["error"])

    if run.get("artifacts"):
        st.markdown("#### Downloads")
    for artifact in run.get("artifacts", []):
        downloaded = download_artifact(artifact["artifact_id"], st.session_state.session_id)
        if downloaded:
            content, media_type = downloaded
            if media_type == "image/png":
                st.image(content, caption=artifact["name"], use_container_width=True)
            st.download_button(
                label=f"Download {artifact['name']}",
                data=content,
                file_name=artifact["name"],
                mime=media_type,
                key=f"artifact-{run['task_id']}-{artifact['artifact_id']}",
            )
