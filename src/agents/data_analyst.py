"""Constrained tool-using data analysis and chart agent."""

import json
from io import BytesIO
from typing import Literal

import pandas as pd
from langchain_core.tools import tool
from matplotlib.figure import Figure

from src.agents.base import AgentContext, ToolCallingAgent
from src.db.artifact_store import save_artifact
from src.db.dataset_store import list_datasets, load_dataframe
from src.db.evidence_store import add_evidence


def _validate_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Unknown columns: {', '.join(missing)}")


class DataAnalystAgent(ToolCallingAgent):
    name = "data_analyst"
    system_prompt = (
        "Analyze workspace datasets with typed statistical tools. Inspect schema before analysis, "
        "perform calculations rather than guessing, create a chart when it adds value, and report limits."
    )

    def build_tools(self, context: AgentContext):
        async def record(content: str, source: str, metadata: dict) -> str:
            evidence_id = await add_evidence(
                task_id=context.task_id,
                session_id=context.session_id,
                agent=self.name,
                content=content,
                source=source,
                confidence=0.95,
                metadata=metadata,
            )
            context.evidence_ids.append(evidence_id)
            return evidence_id

        @tool
        async def list_workspace_datasets() -> list[dict]:
            """List datasets available in this workspace with IDs, columns, and row counts."""
            return await list_datasets(context.session_id)

        @tool
        async def inspect_dataset(dataset_id: str, preview_rows: int = 5) -> dict:
            """Inspect a dataset's schema, missing values, and a small row preview."""
            frame, metadata = await load_dataframe(dataset_id, context.session_id)
            preview_rows = max(1, min(preview_rows, 20))
            result = {
                "dataset": metadata,
                "missing_values": frame.isna().sum().to_dict(),
                "preview": json.loads(frame.head(preview_rows).to_json(orient="records")),
            }
            await record(
                json.dumps(result, default=str), metadata["filename"], {"operation": "inspect"}
            )
            return result

        @tool
        async def analyze_dataset(
            dataset_id: str,
            operation: Literal["describe", "missing", "correlation", "value_counts"],
            columns: list[str] | None = None,
        ) -> dict:
            """Run a safe descriptive, missing-value, correlation, or value-count analysis."""
            frame, metadata = await load_dataframe(dataset_id, context.session_id)
            selected = columns or list(frame.columns)
            _validate_columns(frame, selected)
            subset = frame[selected]
            if operation == "describe":
                output = subset.describe(include="all").fillna("").to_dict()
            elif operation == "missing":
                output = subset.isna().sum().to_dict()
            elif operation == "correlation":
                output = subset.select_dtypes(include="number").corr().fillna(0).to_dict()
            else:
                output = {
                    column: subset[column].value_counts(dropna=False).head(20).to_dict()
                    for column in selected[:5]
                }
            serialized = json.loads(json.dumps(output, default=str))
            evidence_id = await record(
                json.dumps(serialized),
                metadata["filename"],
                {"operation": operation, "columns": selected},
            )
            return {"evidence_id": evidence_id, "result": serialized}

        @tool
        async def aggregate_dataset(
            dataset_id: str,
            group_by: str,
            metric: str,
            aggregation: Literal["sum", "mean", "count", "min", "max"],
        ) -> dict:
            """Group a dataset by one column and aggregate another column."""
            frame, metadata = await load_dataframe(dataset_id, context.session_id)
            _validate_columns(frame, [group_by, metric])
            grouped = frame.groupby(group_by, dropna=False)[metric].agg(aggregation).head(100)
            output = {str(key): value for key, value in grouped.to_dict().items()}
            serialized = json.loads(json.dumps(output, default=str))
            evidence_id = await record(
                json.dumps(serialized),
                metadata["filename"],
                {"operation": aggregation, "group_by": group_by, "metric": metric},
            )
            return {"evidence_id": evidence_id, "result": serialized}

        @tool
        async def create_chart(
            dataset_id: str,
            chart_type: Literal["bar", "line", "scatter", "histogram"],
            x: str,
            y: str | None = None,
            title: str = "Data analysis",
        ) -> dict:
            """Create and save a bounded PNG chart from a workspace dataset."""
            frame, metadata = await load_dataframe(dataset_id, context.session_id)
            required = [x] + ([y] if y else [])
            _validate_columns(frame, required)
            sample = frame[required].dropna().head(1000)
            figure = Figure(figsize=(9, 5))
            axis = figure.subplots()
            if chart_type == "histogram":
                axis.hist(sample[x], bins=20)
                axis.set_xlabel(x)
            elif chart_type == "scatter":
                if not y:
                    raise ValueError("Scatter charts require y")
                axis.scatter(sample[x], sample[y], alpha=0.7)
                axis.set_xlabel(x)
                axis.set_ylabel(y)
            else:
                if not y:
                    raise ValueError(f"{chart_type} charts require y")
                limited = sample.head(50)
                if chart_type == "bar":
                    axis.bar(limited[x].astype(str), limited[y])
                    axis.tick_params(axis="x", rotation=45)
                else:
                    axis.plot(limited[x], limited[y], marker="o")
                axis.set_xlabel(x)
                axis.set_ylabel(y)
            axis.set_title(title)
            figure.tight_layout()
            buffer = BytesIO()
            figure.savefig(buffer, format="png", dpi=120)
            artifact_id = await save_artifact(
                task_id=context.task_id,
                session_id=context.session_id,
                name="analysis-chart.png",
                media_type="image/png",
                content=buffer.getvalue(),
            )
            context.artifact_ids.append(artifact_id)
            return {"artifact_id": artifact_id, "name": "analysis-chart.png"}

        return [
            list_workspace_datasets,
            inspect_dataset,
            analyze_dataset,
            aggregate_dataset,
            create_chart,
        ]

    async def run(self, context: AgentContext):
        """Execute a safe baseline analysis without relying on model-selected tool calls."""
        from src.models.agent import AgentResult

        available = await list_datasets(context.session_id)
        if not available:
            return AgentResult(
                agent=self.name,
                instruction=context.instruction,
                summary="No dataset is available in this workspace.",
                error="dataset_not_found",
            )

        selected = next(
            (
                item
                for item in available
                if str(item["dataset_id"]) in context.instruction
                or str(item["dataset_id"]) in context.objective
            ),
            available[0],
        )
        frame, metadata = await load_dataframe(selected["dataset_id"], context.session_id)
        working = frame.copy()
        lower_columns = {str(column).lower(): str(column) for column in working.columns}
        revenue_column = lower_columns.get("revenue")
        cost_column = lower_columns.get("cost")
        if revenue_column and cost_column:
            working["profit"] = (
                pd.to_numeric(working[revenue_column], errors="coerce")
                - pd.to_numeric(working[cost_column], errors="coerce")
            )

        numeric_columns = [str(column) for column in working.select_dtypes(include="number").columns]
        categorical_columns = [
            str(column)
            for column in working.columns
            if column not in numeric_columns and 1 < working[column].nunique(dropna=False) <= 50
        ]
        numeric_summary = {
            column: {
                "count": int(working[column].count()),
                "sum": round(float(working[column].sum()), 4),
                "mean": round(float(working[column].mean()), 4),
                "min": round(float(working[column].min()), 4),
                "max": round(float(working[column].max()), 4),
            }
            for column in numeric_columns
        }
        grouped_results: dict[str, dict] = {}
        performers: dict[str, dict] = {}
        for group_by in (categorical_columns[:2] if numeric_columns else []):
            grouped = (
                working.groupby(group_by, dropna=False)[numeric_columns]
                .sum(numeric_only=True)
                .round(4)
                .head(30)
            )
            grouped_results[group_by] = json.loads(grouped.to_json(orient="index"))
            primary_metric = "profit" if "profit" in numeric_columns else numeric_columns[0]
            metric_values = grouped[primary_metric]
            performers[group_by] = {
                "metric": primary_metric,
                "top": {
                    "group": str(metric_values.idxmax()),
                    "value": round(float(metric_values.max()), 4),
                },
                "bottom": {
                    "group": str(metric_values.idxmin()),
                    "value": round(float(metric_values.min()), 4),
                },
            }

        analysis = {
            "dataset": {
                "dataset_id": selected["dataset_id"],
                "filename": metadata["filename"],
                "rows": len(working),
                "columns": list(working.columns),
            },
            "missing_values": {str(key): int(value) for key, value in working.isna().sum().items()},
            "numeric_summary": numeric_summary,
            "grouped_sums": grouped_results,
            "top_and_bottom_performers": performers,
        }
        evidence_id = await add_evidence(
            task_id=context.task_id,
            session_id=context.session_id,
            agent=self.name,
            content=json.dumps(analysis, default=str),
            source=metadata["filename"],
            confidence=0.98,
            metadata={"operation": "automatic_baseline_analysis"},
        )
        context.evidence_ids.append(evidence_id)

        tool_calls = 3
        if categorical_columns and numeric_columns:
            group_by = categorical_columns[0]
            metric = "profit" if "profit" in numeric_columns else numeric_columns[0]
            chart_data = (
                working.groupby(group_by, dropna=False)[metric]
                .sum()
                .sort_values(ascending=False)
                .head(20)
            )
            figure = Figure(figsize=(9, 5))
            axis = figure.subplots()
            axis.bar(chart_data.index.astype(str), chart_data.values)
            axis.set_xlabel(group_by)
            axis.set_ylabel(metric)
            axis.set_title(f"{metric.title()} by {group_by.title()}")
            axis.tick_params(axis="x", rotation=45)
            figure.tight_layout()
            buffer = BytesIO()
            figure.savefig(buffer, format="png", dpi=120)
            artifact_id = await save_artifact(
                task_id=context.task_id,
                session_id=context.session_id,
                name="analysis-chart.png",
                media_type="image/png",
                content=buffer.getvalue(),
            )
            context.artifact_ids.append(artifact_id)
            tool_calls += 1

        lines = [
            f"## Dataset analysis: {metadata['filename']}",
            f"Analyzed {len(working)} rows and {len(working.columns)} columns.",
            "",
            "### Numeric summary",
        ]
        for column, values in numeric_summary.items():
            lines.append(
                f"- **{column}**: total {values['sum']}, average {values['mean']}, "
                f"range {values['min']} to {values['max']}"
            )
        for group_by, values in grouped_results.items():
            lines.extend(
                ["", f"### Totals by {group_by}", "```json", json.dumps(values, indent=2), "```"]
            )
            ranking = performers[group_by]
            lines.append(
                f"Top by {ranking['metric']}: {ranking['top']['group']} "
                f"({ranking['top']['value']}); bottom: {ranking['bottom']['group']} "
                f"({ranking['bottom']['value']})."
            )
        return AgentResult(
            agent=self.name,
            instruction=context.instruction,
            summary="\n".join(lines),
            evidence_ids=context.evidence_ids,
            tool_calls=tool_calls,
        )


data_analyst = DataAnalystAgent()
