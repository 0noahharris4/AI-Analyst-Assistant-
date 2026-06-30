
from __future__ import annotations
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional imports
try:
    from ydata_profiling import ProfileReport
    HAS_PROFILE = True
except Exception:
    HAS_PROFILE = False

try:
    from scipy.stats import skew, kurtosis
except Exception:
    skew = lambda x: float("nan")  # type: ignore
    kurtosis = lambda x: float("nan")  # type: ignore


# ---------------------------------------------------------------------
# Configuration and metadata models
# ---------------------------------------------------------------------

@dataclass
class AnalyzerConfig:
    """
    Configuration options for the DataAnalyzer pipeline.
    """
    numeric_conversion_threshold: float = 0.5
    datetime_conversion_threshold: float = 0.5
    top_correlation_pairs: int = 10
    multicollinearity_threshold: float = 0.85
    top_categorical_values: int = 10
    time_series_resample_freqs: Tuple[str, ...] = ("D", "W", "M")
    spike_rolling_window: int = 7
    spike_threshold_std: float = 3.0
    generate_profile: bool = False          # ydata_profiling
    profile_minimal: bool = True            # Profiling config
    sample_for_inconsistent: int = 100      # sample size for format detection
    problematic_sample_n: int = 5           # samples returned for problematic cols


@dataclass
class ColumnMetadata:
    """
    Standardized metadata for a single column.
    """
    name: str
    dtype: str
    inferred_type: str
    missing_count: int
    missing_pct: float
    unique_count: int
    is_constant: bool
    is_fully_empty: bool
    inferred_roles: List[str]  # e.g., ["numeric", "identifier", "categorical"]
    warnings: List[str]


@dataclass
class InitialAnalysisResult:
    head: pd.DataFrame
    shape: Tuple[int, int]
    dtypes: pd.Series
    problematic_columns: Dict[str, List[str]]
    missing_counts: pd.Series
    fully_empty: List[str]
    more_than_50_missing: List[str]
    duplicate_rows: int
    inconsistent_formats: Dict[str, List[str]]
    column_metadata: Dict[str, ColumnMetadata]


@dataclass
class CleaningSummary:
    before_shape: Tuple[int, int]
    after_shape: Tuple[int, int]
    dropped_fully_empty_cols: List[str]
    dropped_duplicates: int
    note: str


@dataclass
class NumericStatsResult:
    stats: pd.DataFrame  # Indexed by column with mean/median/std/etc.


@dataclass
class CategoricalStatsResult:
    stats: Dict[str, Dict[str, Any]]  # col -> {unique, top_categories}


@dataclass
class CorrelationResult:
    corr_matrix: Optional[pd.DataFrame]
    top_pairs: List[Tuple[str, str, float]]
    multicollinear_pairs: List[Tuple[str, str, float]]


@dataclass
class TimeSeriesResult:
    datetime_column: Optional[str]
    resampled: Dict[str, Optional[pd.DataFrame]]


@dataclass
class ProblematicColumnsResult:
    problematic: Dict[str, Dict[str, Any]]


@dataclass
class ProfileResult:
    html: Optional[str]


@dataclass
class FullAnalysisSummary:
    """
    Consolidated, AI/JSON-friendly summary of all analysis steps.
    """
    shape: Tuple[int, int]
    initial: Optional[Dict[str, Any]]
    cleaning: Optional[Dict[str, Any]]
    numeric_stats: Optional[Dict[str, Any]]
    categorical_stats: Optional[Dict[str, Any]]
    correlation: Optional[Dict[str, Any]]
    time_series: Optional[Dict[str, Any]]
    problematic_columns: Optional[Dict[str, Any]]
    profile_html: Optional[str]


# ---------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------


class DataAnalyzer:
    """
    Stateful analyzer that runs EDA, cleaning, and profiling on a DataFrame.
    """

    def __init__(self, df: pd.DataFrame, config: Optional[AnalyzerConfig] = None):
        self.original_df = df.copy()
        self.df = df.copy()
        self.config = config or AnalyzerConfig()

        # Internal storage for results
        self._initial_result: Optional[InitialAnalysisResult] = None
        self._cleaning_summary: Optional[CleaningSummary] = None
        self._numeric_stats: Optional[NumericStatsResult] = None
        self._categorical_stats: Optional[CategoricalStatsResult] = None
        self._correlation: Optional[CorrelationResult] = None
        self._time_series: Optional[TimeSeriesResult] = None
        self._problematic_columns: Optional[ProblematicColumnsResult] = None
        self._profile: Optional[ProfileResult] = None

    # -----------------------------------------------------------------
    # Public pipeline-style API
    # -----------------------------------------------------------------

    def run_full_analysis(self) -> FullAnalysisSummary:
        """
        Convenience method to run the most common analysis steps.
        """
        self.initial_analysis()
        self.clean()
        self.compute_numeric_stats()
        self.compute_categorical_stats()
        self.compute_correlation()
        self.time_series_analysis()
        self.detect_problematic_columns()
        if self.config.generate_profile:
            self.generate_profile()

        return self.build_full_summary()

    # -----------------------------------------------------------------
    # Step 1: Initial analysis
    # -----------------------------------------------------------------

    def initial_analysis(self) -> InitialAnalysisResult:
        df = self.df

        head = df.head()
        shape = df.shape
        dtypes = df.dtypes

        # Problematic columns (fully empty, constant)
        problematic = {}
        fully_empty_cols = []
        constant_cols = []

        for col in df.columns:
            ser = df[col]
            if ser.isna().all():
                fully_empty_cols.append(col)
            if ser.nunique(dropna=False) == 1:
                constant_cols.append(col)

        if fully_empty_cols:
            problematic["fully_empty"] = fully_empty_cols
        if constant_cols:
            problematic["constant"] = constant_cols

        # Missing values
        missing = df.isna().sum().sort_values(ascending=False)
        fully_empty = list(missing[missing == df.shape[0]].index)
        more50 = list(missing[missing > 0.5 * df.shape[0]].index)

        # Duplicates
        duplicates = int(df.duplicated().sum())

        # Basic inconsistent format detection for object columns
        inconsistent = {}
        sample_n = self.config.sample_for_inconsistent

        for col in df.select_dtypes(include=["object"]).columns:
            sample = df[col].dropna().astype(str).head(sample_n)

            # date-like
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().sum() > 0:
                inconsistent.setdefault("date_like", []).append(col)

            # currency-like (robust to ArrowStringArray/pyarrow issues)
            import re
            currency_regex = re.compile(r"[\$\£\€]")
            try:
                # Convert to list to avoid ArrowStringArray issues
                if any(currency_regex.search(val) for val in sample.tolist()):
                    inconsistent.setdefault("currency_like", []).append(col)
            except Exception:
                # Fallback: skip this column if error
                pass

            # boolean-like
            bool_like = sample.str.lower().isin(
                ["true", "false", "yes", "no", "0", "1"]
            )
            if bool_like.sum() > 0:
                inconsistent.setdefault("bool_like", []).append(col)

        column_metadata = self._build_column_metadata(df, missing)

        result = InitialAnalysisResult(
            head=head,
            shape=shape,
            dtypes=dtypes,
            problematic_columns=problematic,
            missing_counts=missing,
            fully_empty=fully_empty,
            more_than_50_missing=more50,
            duplicate_rows=duplicates,
            inconsistent_formats=inconsistent,
            column_metadata=column_metadata,
        )

        self._initial_result = result
        return result

    # -----------------------------------------------------------------
    # Step 2: Cleaning
    # -----------------------------------------------------------------

    def clean(self) -> CleaningSummary:
        df = self.df
        before_shape = df.shape

        # Determine fully empty columns based on current df
        fully_empty_cols = [c for c in df.columns if df[c].isna().all()]
        df = df.dropna(axis=1, how="all")

        # Drop duplicates
        before_dup = df.shape[0]
        df = df.drop_duplicates()
        dropped_duplicates = before_dup - df.shape[0]

        # Trim whitespace in object columns
        obj_cols = df.select_dtypes(include=["object"]).columns
        for c in obj_cols:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .replace({"nan": np.nan})  # convert literal "nan" back to NaN
            )

        # Convert numeric-like strings to numeric
        thresh = self.config.numeric_conversion_threshold
        for c in df.columns:
            if df[c].dtype == object:
                coerced = pd.to_numeric(
                    df[c].astype(str).str.replace(r"[\$,]", "", regex=True),
                    errors="coerce",
                )
                non_null = coerced.notna().sum()
                if df.shape[0] > 0 and non_null / df.shape[0] > thresh:
                    df[c] = coerced

        # Parse datetime-like columns
        dt_thresh = self.config.datetime_conversion_threshold
        for c in df.columns:
            if df[c].dtype == object:
                dt = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
                non_null = dt.notna().sum()
                if df.shape[0] > 0 and non_null / df.shape[0] > dt_thresh:
                    df[c] = dt

        after_shape = df.shape
        note = (
            "Trimmed whitespace, converted numeric-like strings, parsed "
            "date-like columns, dropped duplicates and fully-empty columns."
        )

        summary = CleaningSummary(
            before_shape=before_shape,
            after_shape=after_shape,
            dropped_fully_empty_cols=fully_empty_cols,
            dropped_duplicates=dropped_duplicates,
            note=note,
        )

        self.df = df
        self._cleaning_summary = summary
        return summary

    # -----------------------------------------------------------------
    # Step 3: Numeric stats
    # -----------------------------------------------------------------

    def compute_numeric_stats(self) -> NumericStatsResult:
        num = self.df.select_dtypes(include=[np.number])
        if num.shape[1] == 0:
            stats_df = pd.DataFrame()
        else:
            stats_df = num.agg(["mean", "median", "std", "min", "max", "var"]).T
            stats_df["skew"] = num.apply(lambda x: skew(x.dropna()))
            stats_df["kurtosis"] = num.apply(lambda x: kurtosis(x.dropna()))
            stats_df = stats_df.sort_values("var", ascending=False)

        result = NumericStatsResult(stats=stats_df)
        self._numeric_stats = result
        return result

    # -----------------------------------------------------------------
    # Step 4: Categorical stats
    # -----------------------------------------------------------------

    def compute_categorical_stats(self) -> CategoricalStatsResult:
        cat = self.df.select_dtypes(include=["object", "category"])
        out: Dict[str, Dict[str, Any]] = {}
        top_n = self.config.top_categorical_values

        for c in cat.columns:
            vc = self.df[c].value_counts(dropna=False)
            out[c] = {
                "unique": int(self.df[c].nunique(dropna=True)),
                "top_categories": vc.head(top_n).to_dict(),
            }

        result = CategoricalStatsResult(stats=out)
        self._categorical_stats = result
        return result

    # -----------------------------------------------------------------
    # Step 5: Distribution plots (kept as pure helpers)
    # -----------------------------------------------------------------

    def create_distribution_plots(
        self,
    ) -> Tuple[Dict[str, plt.Figure], Dict[str, plt.Figure]]:
        """
        Create histogram+boxplot for numeric columns and bar charts for categorical columns.
        Returns:
            numeric_plots: {col: fig}
            categorical_plots: {col: fig}
        """
        df = self.df

        num = df.select_dtypes(include=[np.number])
        numeric_plots: Dict[str, plt.Figure] = {}
        for c in num.columns:
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            sns.histplot(df[c].dropna(), ax=axes[0], kde=True)
            axes[0].set_title(f"Histogram {c}")
            sns.boxplot(x=df[c].dropna(), ax=axes[1])
            axes[1].set_title(f"Boxplot {c}")
            plt.tight_layout()
            numeric_plots[c] = fig

        cat = df.select_dtypes(include=["object", "category"])
        categorical_plots: Dict[str, plt.Figure] = {}
        max_bars = 12
        for c in cat.columns:
            fig, ax = plt.subplots(figsize=(6, 3))
            vc = df[c].value_counts(dropna=False)
            if vc.size > max_bars:
                top = vc.head(max_bars - 1)
                others = vc.iloc[max_bars - 1 :].sum()
                plot_series = pd.concat([top, pd.Series({'Other': others})])
            else:
                plot_series = vc.head(max_bars)

            plot_series.plot.bar(ax=ax, color=sns.color_palette('pastel'))
            ax.set_title(f"Value counts {c}")
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            for label in ax.get_xticklabels():
                label.set_ha('right')
            plt.tight_layout()
            categorical_plots[c] = fig

        return numeric_plots, categorical_plots

    # -----------------------------------------------------------------
    # Step 6: Correlation analysis
    # -----------------------------------------------------------------

    def compute_correlation(self) -> CorrelationResult:
        num = self.df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            result = CorrelationResult(
                corr_matrix=None,
                top_pairs=[],
                multicollinear_pairs=[],
            )
            self._correlation = result
            return result

        corr = num.corr()
        corr_pairs: List[Tuple[str, str, float]] = []
        cols = list(corr.columns)

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                corr_pairs.append((a, b, float(corr.iloc[i, j])))

        corr_pairs = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
        top_n = self.config.top_correlation_pairs
        top_pairs = corr_pairs[:top_n]

        # Multicollinearity detection
        threshold = self.config.multicollinearity_threshold
        multicollinear_pairs = [p for p in top_pairs if abs(p[2]) > threshold]

        result = CorrelationResult(
            corr_matrix=corr,
            top_pairs=top_pairs,
            multicollinear_pairs=multicollinear_pairs,
        )

        self._correlation = result
        return result

    # -----------------------------------------------------------------
    # Step 7: Time series analysis
    # -----------------------------------------------------------------

    def time_series_analysis(self) -> TimeSeriesResult:
        df = self.df
        dcols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
        resampled: Dict[str, Optional[pd.DataFrame]] = {}

        if not dcols:
            result = TimeSeriesResult(
                datetime_column=None,
                resampled=resampled,
            )
            self._time_series = result
            return result

        dtcol = dcols[0]
        ts = df.set_index(dtcol)

        # choose numeric columns for resample
        num_cols = ts.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            result = TimeSeriesResult(
                datetime_column=dtcol,
                resampled=resampled,
            )
            self._time_series = result
            return result

        # Resampling
        for freq in self.config.time_series_resample_freqs:
            try:
                resampled[freq] = ts[num_cols].resample(freq).mean()
            except Exception:
                resampled[freq] = None

        # No spike detection: return only resampled results
        result = TimeSeriesResult(
            datetime_column=dtcol,
            resampled=resampled,
        )
        self._time_series = result
        return result

    # -----------------------------------------------------------------
    # Step 8: Profile report (optional)
    # -----------------------------------------------------------------

    def generate_profile(self) -> ProfileResult:
        if not HAS_PROFILE:
            warnings.warn("ydata_profiling is not installed; profile HTML will be None.")
            result = ProfileResult(html=None)
            self._profile = result
            return result

        prof = ProfileReport(self.df, minimal=self.config.profile_minimal)
        html = prof.to_html()
        result = ProfileResult(html=html)
        self._profile = result
        return result

    # -----------------------------------------------------------------
    # Step 9: Problematic columns & sanitization
    # -----------------------------------------------------------------

    def detect_problematic_columns(self) -> ProblematicColumnsResult:
        """
        Detect columns that may be incompatible with Arrow/Streamlit (contain non-primitive objects).
        Returns a dict mapping column -> {'type': str, 'samples': [..]}
        """
        problematic: Dict[str, Dict[str, Any]] = {}
        primitive_types = (str, int, float, bool)
        sample_n = self.config.problematic_sample_n

        for col in self.df.columns:
            values = self.df[col].dropna().head(100)
            bad_samples = []
            bad_type: Optional[str] = None
            for v in values:
                try:
                    if (
                        isinstance(v, primitive_types)
                        or pd.api.types.is_scalar(v)
                        and not isinstance(v, (list, dict, set, bytes, bytearray))
                    ):
                        continue
                except Exception:
                    pass
                # mark as problematic
                bad_samples.append(v)
                if bad_type is None:
                    bad_type = type(v).__name__
                if len(bad_samples) >= sample_n:
                    break
            if bad_samples:
                problematic[col] = {
                    "type": bad_type or "object",
                    "samples": bad_samples,
                }

        result = ProblematicColumnsResult(problematic=problematic)
        self._problematic_columns = result
        return result

    def sanitize_for_display(self) -> pd.DataFrame:
        """
        Return a copy of df where non-primitive objects are converted to readable strings
        for display (e.g., Streamlit/pyarrow).
        """
        out = self.df.copy()
        for col in out.columns:
            try:
                col_vals = out[col]
                mask = col_vals.apply(
                    lambda x: not isinstance(
                        x,
                        (
                            str,
                            int,
                            float,
                            bool,
                            type(None),
                            np.integer,
                            np.floating,
                            np.bool_,
                        ),
                    )
                )
                if mask.any():
                    out[col] = col_vals.apply(
                        lambda x: x
                        if isinstance(
                            x,
                            (str, int, float, bool, type(None)),
                        )
                        else repr(x)
                    )
            except Exception:
                # fallback: convert entire column to string
                try:
                    out[col] = out[col].astype(str)
                except Exception:
                    out[col] = out[col].apply(lambda x: repr(x))
        return out

    # -----------------------------------------------------------------
    # Step 10: AI-friendly summary / prompt building
    # -----------------------------------------------------------------

    def build_full_summary(self) -> FullAnalysisSummary:
        """
        Build a consolidated, JSON/LLM-friendly summary of the analysis.
        Does not trigger new computation; uses stored results.
        """
        shape = tuple(self.df.shape)

        initial_dict: Optional[Dict[str, Any]] = None
        if self._initial_result is not None:
            ir = self._initial_result
            initial_dict = {
                "shape": ir.shape,
                "dtypes": ir.dtypes.astype(str).to_dict(),
                "problematic_columns": ir.problematic_columns,
                "missing_counts": ir.missing_counts.to_dict(),
                "fully_empty": ir.fully_empty,
                "more_than_50_missing": ir.more_than_50_missing,
                "duplicate_rows": ir.duplicate_rows,
                "inconsistent_formats": ir.inconsistent_formats,
                "column_metadata": {
                    name: asdict(meta) for name, meta in ir.column_metadata.items()
                },
            }

        cleaning_dict: Optional[Dict[str, Any]] = None
        if self._cleaning_summary is not None:
            cleaning_dict = asdict(self._cleaning_summary)

        numeric_dict: Optional[Dict[str, Any]] = None
        if self._numeric_stats is not None:
            # Include full stats, but also highlight top variance columns
            stats_df = self._numeric_stats.stats
            if not stats_df.empty:
                numeric_dict = {
                    "stats": stats_df.to_dict(orient="index"),
                    "top_variances": stats_df["var"]
                    .sort_values(ascending=False)
                    .head(10)
                    .to_dict(),
                }
            else:
                numeric_dict = {"stats": {}, "top_variances": {}}

        categorical_dict: Optional[Dict[str, Any]] = None
        if self._categorical_stats is not None:
            categorical_dict = self._categorical_stats.stats

        correlation_dict: Optional[Dict[str, Any]] = None
        if self._correlation is not None:
            corr_res = self._correlation
            correlation_dict = {
                "top_pairs": [
                    {"col_a": a, "col_b": b, "corr": float(c)}
                    for (a, b, c) in corr_res.top_pairs
                ],
                "multicollinear_pairs": [
                    {"col_a": a, "col_b": b, "corr": float(c)}
                    for (a, b, c) in corr_res.multicollinear_pairs
                ],
            }

        ts_dict: Optional[Dict[str, Any]] = None
        if self._time_series is not None:
            ts_res = self._time_series
            ts_dict = {
                "datetime_column": ts_res.datetime_column,
                # resampled is potentially large; include only shapes
                "resampled_shapes": {
                    freq: None
                    if df is None
                    else {"rows": df.shape[0], "cols": df.shape[1]}
                    for freq, df in ts_res.resampled.items()
                },
            }

        problematic_dict: Optional[Dict[str, Any]] = None
        if self._problematic_columns is not None:
            problematic_dict = self._problematic_columns.problematic

        profile_html = None
        if self._profile is not None:
            profile_html = self._profile.html

        summary = FullAnalysisSummary(
            shape=shape,
            initial=initial_dict,
            cleaning=cleaning_dict,
            numeric_stats=numeric_dict,
            categorical_stats=categorical_dict,
            correlation=correlation_dict,
            time_series=ts_dict,
            problematic_columns=problematic_dict,
            profile_html=profile_html,
        )
        return summary

    def build_ai_prompt(self) -> str:
        """
        Build a succinct text summary for LLMs from the current stored results.
        """
        summary = self.build_full_summary()

        parts = []
        parts.append(f"Dataset shape: {summary.shape}")

        # Missing summary (top 10)
        if summary.initial and "missing_counts" in summary.initial:
            missing = summary.initial["missing_counts"]
            missing_top = dict(
                sorted(missing.items(), key=lambda x: x[1], reverse=True)[:10]
            )
            parts.append(f"Missing summary (top 10):\n{missing_top}")

        # Correlations
        if summary.correlation and "top_pairs" in summary.correlation:
            parts.append(
                f"Top correlations: {summary.correlation.get('top_pairs', [])}"
            )

        # Variances
        if summary.numeric_stats and "top_variances" in summary.numeric_stats:
            parts.append(
                f"Top variances: {summary.numeric_stats.get('top_variances', {})}"
            )

        # Outlier summary placeholder (not deeply implemented yet)
        parts.append("Outlier summary (per numeric column): {}")

        # Categorical sample
        if summary.categorical_stats:
            sample_cats = {
                k: v
                for i, (k, v) in enumerate(summary.categorical_stats.items())
                if i < 5
            }
            parts.append(f"Categorical distributions (sample): {sample_cats}")

        # Numeric stats sample
        if summary.numeric_stats and "stats" in summary.numeric_stats:
            stats = summary.numeric_stats["stats"]
            # take head-like sample of 5 cols
            sample_keys = list(stats.keys())[:5]
            numeric_sample = {k: stats[k] for k in sample_keys}
            parts.append(f"Numeric stats sample: {numeric_sample}")

        return "\n\n".join(parts)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _build_column_metadata(
        self, df: pd.DataFrame, missing_counts: pd.Series
    ) -> Dict[str, ColumnMetadata]:
        metadata: Dict[str, ColumnMetadata] = {}
        n_rows = df.shape[0] if df.shape[0] > 0 else 1

        for col in df.columns:
            ser = df[col]
            dtype_str = str(ser.dtype)
            missing = int(missing_counts.get(col, ser.isna().sum()))
            missing_pct = missing / n_rows
            unique_count = int(ser.nunique(dropna=True))
            is_constant = unique_count <= 1
            is_fully_empty = missing == n_rows

            inferred_roles: List[str] = []
            warnings_list: List[str] = []

            # Infer types and roles
            if pd.api.types.is_numeric_dtype(ser):
                inferred_type = "numeric"
                inferred_roles.append("numeric")
            elif pd.api.types.is_datetime64_any_dtype(ser):
                inferred_type = "datetime"
                inferred_roles.append("datetime")
            elif pd.api.types.is_bool_dtype(ser):
                inferred_type = "boolean"
                inferred_roles.append("categorical")
            else:
                inferred_type = "categorical"
                inferred_roles.append("categorical")

            if unique_count == n_rows and n_rows > 1:
                inferred_roles.append("identifier")

            if missing_pct > 0.5:
                warnings_list.append("High missing percentage (>50%).")
            if is_constant:
                warnings_list.append("Column is constant.")
            if is_fully_empty:
                warnings_list.append("Column is fully empty.")

            metadata[col] = ColumnMetadata(
                name=col,
                dtype=dtype_str,
                inferred_type=inferred_type,
                missing_count=missing,
                missing_pct=missing_pct,
                unique_count=unique_count,
                is_constant=is_constant,
                is_fully_empty=is_fully_empty,
                inferred_roles=inferred_roles,
                warnings=warnings_list,
            )

        return metadata


# ---------------------------------------------------------------------
# Compatibility wrapper functions for top-level API expected by Data_Agent.py
# ---------------------------------------------------------------------


def initial_analysis(df: pd.DataFrame) -> dict:
    analyzer = DataAnalyzer(df)
    res = analyzer.initial_analysis()
    return {
        "head": res.head,
        "shape": res.shape,
        "dtypes": res.dtypes,
        "problematic": res.problematic_columns,
        "missing": res.missing_counts,
        "fully_empty": res.fully_empty,
        "more50": res.more_than_50_missing,
        "duplicates": res.duplicate_rows,
        "inconsistent": res.inconsistent_formats,
    }


def clean_data(df: pd.DataFrame) -> tuple:
    analyzer = DataAnalyzer(df.copy())
    summary = analyzer.clean()
    return analyzer.df, asdict(summary)


def numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    analyzer = DataAnalyzer(df)
    res = analyzer.compute_numeric_stats()
    return res.stats


def categorical_stats(df: pd.DataFrame, top_n: int = 10) -> dict:
    cfg = AnalyzerConfig(top_categorical_values=top_n)
    analyzer = DataAnalyzer(df, cfg)
    res = analyzer.compute_categorical_stats()
    return res.stats


def distribution_plots(df: pd.DataFrame):
    analyzer = DataAnalyzer(df)
    return analyzer.create_distribution_plots()


def correlation_analysis(df: pd.DataFrame) -> dict:
    analyzer = DataAnalyzer(df)
    res = analyzer.compute_correlation()
    return {
        "corr": res.corr_matrix,
        "top_pairs": res.top_pairs,
        "multicollinear": res.multicollinear_pairs,
    }


def time_series_analysis(df: pd.DataFrame) -> dict:
    analyzer = DataAnalyzer(df)
    res = analyzer.time_series_analysis()
    return {
        "datetime_column": res.datetime_column,
        "resampled": res.resampled,
    }


def generate_profile(df: pd.DataFrame) -> str:
    analyzer = DataAnalyzer(df)
    res = analyzer.generate_profile()
    return res.html


def build_ai_prompt(summary: dict) -> str:
    """
    Build a detailed prompt for the LLM to generate an expert-level data analysis summary.
    """
    parts = []
    parts.append("You are an expert data analyst. Given the following dataset summary, provide a comprehensive, insightful, and actionable analysis. Do NOT include generic headers or introductions such as 'Presentation to Business Stakeholders'. Focus on clear, concise, and professional insights only. Your response should include:")
    parts.append("- Key findings and trends in the data (including numeric and categorical features)")
    parts.append("- Notable correlations and relationships between variables")
    parts.append("- Any detected anomalies, outliers, or data quality issues")
    parts.append("- Actionable recommendations or next steps for further analysis or business decisions")
    parts.append("")
    parts.append(f"Dataset shape: {summary.get('shape')}")
    missing = summary.get('missing')
    if missing is not None:
        try:
            missing_top = missing.sort_values(ascending=False)[:10].to_dict()
        except Exception:
            missing_top = dict(list(missing.items())[:10]) if isinstance(missing, dict) else {}
        parts.append(f"Missing summary (top 10): {missing_top}")
    if "correlation_top" in summary:
        parts.append(f"Top correlations: {summary.get('correlation_top')}")
    if "top_variances" in summary:
        parts.append(f"Top variances: {summary.get('top_variances')}")
    parts.append("Outlier summary: {}")
    # Add categorical and numeric stats if available
    if 'categorical_sample' in summary:
        parts.append(f"Categorical distributions (sample): {summary['categorical_sample']}")
    if 'numeric_stats_head' in summary:
        parts.append(f"Numeric stats (sample): {summary['numeric_stats_head']}")
    return "\n\n".join(parts)


def detect_problematic_columns(df: pd.DataFrame) -> dict:
    analyzer = DataAnalyzer(df)
    res = analyzer.detect_problematic_columns()
    return res.problematic


def sanitize_for_display(df: pd.DataFrame) -> pd.DataFrame:
    analyzer = DataAnalyzer(df)
    analyzer.df = df
    return analyzer.sanitize_for_display()


# ---------------------------------------------------------------------
# Example usage (you'd remove or adapt this in production)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Simple demo; replace with real data/loaders in your project
    df_demo = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=10),
            "value": [1, 2, 3, 100, 5, 6, 7, 8, 200, 10],
            "category": ["A", "B", "A", "A", "B", "C", "A", "B", "C", "C"],
            "id": range(10),
        }
    )

    analyzer = DataAnalyzer(df_demo, AnalyzerConfig(generate_profile=False))
    summary = analyzer.run_full_analysis()
    prompt = analyzer.build_ai_prompt()

    print("Full summary shape:", summary.shape)
    print("\nAI prompt:\n", prompt)
