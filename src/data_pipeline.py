"""
Data Processing Pipeline
========================
Robust data collection, preprocessing, validation and quality assurance.
Covers: missing value imputation, outlier detection, normalization, data quality reporting.
Required supporting component as per problem statement.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Expected ranges for each column (domain knowledge) ──────────────────────
VALID_RANGES = {
    # Process file
    'Temperature_C':         (15.0,  80.0),
    'Pressure_Bar':          (0.5,   10.0),
    'Humidity_Percent':      (10.0,  90.0),
    'Motor_Speed_RPM':       (100,   3000),
    'Compression_Force_kN':  (1.0,   80.0),
    'Flow_Rate_LPM':         (1.0,   100.0),
    'Power_Consumption_kW':  (0.5,   80.0),
    'Vibration_mm_s':        (0.0,   15.0),
    # Production file
    'Granulation_Time':      (5,     35),
    'Binder_Amount':         (3.0,   18.0),
    'Drying_Temp':           (40,    90),
    'Drying_Time':           (10,    80),
    'Compression_Force':     (5.0,   25.0),
    'Machine_Speed':         (10,    80),
    'Lubricant_Conc':        (0.1,   2.5),
    'Moisture_Content':      (0.2,   5.0),
    'Tablet_Weight':         (300,   700),
    'Hardness':              (2,     18),
    'Friability':            (0.0,   2.5),
    'Disintegration_Time':   (1.0,   20.0),
    'Dissolution_Rate':      (60.0,  100.0),
    'Content_Uniformity':    (85.0,  115.0),
}


class DataQualityReport:
    """Collects all data quality findings during pipeline execution."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.timestamp = datetime.now().isoformat()
        self.issues = []
        self.stats = {}

    def add_issue(self, severity: str, column: str, description: str, count: int = 0):
        self.issues.append({
            'severity': severity,   # ERROR / WARNING / INFO
            'column': column,
            'description': description,
            'count': count
        })

    def summary(self) -> dict:
        errors   = [i for i in self.issues if i['severity'] == 'ERROR']
        warnings = [i for i in self.issues if i['severity'] == 'WARNING']
        infos    = [i for i in self.issues if i['severity'] == 'INFO']
        return {
            'dataset': self.dataset_name,
            'timestamp': self.timestamp,
            'total_issues': len(self.issues),
            'errors': len(errors),
            'warnings': len(warnings),
            'infos': len(infos),
            'issues': self.issues,
            'stats': self.stats
        }

    def print_report(self):
        s = self.summary()
        print(f"\n{'='*60}")
        print(f"DATA QUALITY REPORT — {self.dataset_name}")
        print(f"Generated: {self.timestamp}")
        print(f"{'='*60}")
        print(f"  Errors  : {s['errors']}")
        print(f"  Warnings: {s['warnings']}")
        print(f"  Infos   : {s['infos']}")
        for issue in self.issues:
            icon = '🔴' if issue['severity'] == 'ERROR' else '🟡' if issue['severity'] == 'WARNING' else '🔵'
            cnt = f" [{issue['count']} rows]" if issue['count'] else ""
            print(f"  {icon} [{issue['severity']}] {issue['column']}: {issue['description']}{cnt}")
        if self.stats:
            print(f"\n  Dataset stats: {self.stats}")
        print()


# ── Step 1: Load ─────────────────────────────────────────────────────────────

def load_raw(path: Path, report: DataQualityReport) -> pd.DataFrame:
    """Load Excel or CSV with error handling."""
    try:
        if str(path).endswith('.csv'):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        report.stats['rows_loaded'] = len(df)
        report.stats['cols_loaded'] = len(df.columns)
        report.add_issue('INFO', 'all', f"Loaded {len(df)} rows × {len(df.columns)} columns")
        return df
    except Exception as e:
        report.add_issue('ERROR', 'file', f"Failed to load: {e}")
        raise


# ── Step 2: Schema validation ─────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame, required_cols: list,
                    report: DataQualityReport) -> pd.DataFrame:
    """Check all required columns exist."""
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        report.add_issue('ERROR', 'schema',
                         f"Missing required columns: {missing_cols}")
    extra_cols = [c for c in df.columns if c not in required_cols]
    if extra_cols:
        report.add_issue('INFO', 'schema',
                         f"Extra columns (ignored): {extra_cols}")
    return df


# ── Step 3: Missing value detection and imputation ────────────────────────────

def handle_missing_values(df: pd.DataFrame,
                           report: DataQualityReport) -> pd.DataFrame:
    """
    Detect and impute missing values.
    Strategy: median imputation for numeric (robust to outliers),
    mode for categorical.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue

        pct = n_missing / len(df) * 100
        severity = 'ERROR' if pct > 30 else 'WARNING' if pct > 10 else 'INFO'
        report.add_issue(severity, col,
                         f"{n_missing} missing values ({pct:.1f}%)",
                         count=n_missing)

        if col in numeric_cols:
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            report.add_issue('INFO', col,
                             f"Imputed with median={fill_val:.3f}")
        else:
            fill_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
            df[col] = df[col].fillna(fill_val)
            report.add_issue('INFO', col,
                             f"Imputed with mode='{fill_val}'")

    total_missing = df.isna().sum().sum()
    if total_missing == 0:
        report.add_issue('INFO', 'all', "No missing values remain after imputation")

    return df


# ── Step 4: Outlier detection (IQR + Z-score combined) ───────────────────────

def detect_and_handle_outliers(df: pd.DataFrame,
                                report: DataQualityReport) -> pd.DataFrame:
    """
    Dual-method outlier detection:
    - IQR method for skewed distributions
    - Z-score for approximately normal distributions
    Outliers are capped (Winsorized) rather than dropped to preserve batch count.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col in ('Time_Minutes',):  # skip index-like columns
            continue

        values = df[col].dropna()

        # IQR method
        Q1, Q3 = values.quantile(0.25), values.quantile(0.75)
        IQR = Q3 - Q1
        lower_iqr = Q1 - 1.5 * IQR
        upper_iqr = Q3 + 1.5 * IQR

        # Z-score method
        z_scores = np.abs((values - values.mean()) / (values.std() + 1e-8))
        z_outliers = (z_scores > 3).sum()

        # Count IQR outliers
        iqr_outliers = ((df[col] < lower_iqr) | (df[col] > upper_iqr)).sum()

        if iqr_outliers > 0 or z_outliers > 0:
            severity = 'WARNING' if iqr_outliers > len(df) * 0.05 else 'INFO'
            report.add_issue(severity, col,
                             f"Outliers detected — IQR: {iqr_outliers}, Z-score: {z_outliers}",
                             count=max(iqr_outliers, z_outliers))

            # Apply domain range cap if available, else use IQR bounds
            if col in VALID_RANGES:
                low, high = VALID_RANGES[col]
            else:
                low, high = lower_iqr, upper_iqr

            # Winsorize — cap at bounds rather than dropping rows
            before_min, before_max = df[col].min(), df[col].max()
            df[col] = df[col].clip(lower=low, upper=high)
            after_min, after_max = df[col].min(), df[col].max()

            if before_min != after_min or before_max != after_max:
                report.add_issue('INFO', col,
                                 f"Capped to [{low}, {high}] — was [{before_min:.2f}, {before_max:.2f}]")

    return df


# ── Step 5: Range validation ──────────────────────────────────────────────────

def validate_ranges(df: pd.DataFrame, report: DataQualityReport) -> pd.DataFrame:
    """Check values fall within domain-valid ranges."""
    for col, (low, high) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        out_of_range = ((df[col] < low) | (df[col] > high)).sum()
        if out_of_range > 0:
            report.add_issue('WARNING', col,
                             f"{out_of_range} values outside valid range [{low}, {high}]",
                             count=out_of_range)
        else:
            report.add_issue('INFO', col, f"All values within valid range [{low}, {high}]")
    return df


# ── Step 6: Normalization ─────────────────────────────────────────────────────

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names — strip whitespace, consistent casing."""
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]
    return df


# ── Step 7: Duplicate detection ───────────────────────────────────────────────

def check_duplicates(df: pd.DataFrame, report: DataQualityReport) -> pd.DataFrame:
    """Remove duplicate rows and report."""
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        report.add_issue('WARNING', 'all',
                         f"{n_dupes} duplicate rows found and removed",
                         count=n_dupes)
        df = df.drop_duplicates()
    else:
        report.add_issue('INFO', 'all', "No duplicate rows found")
    return df


# ── Full pipeline ─────────────────────────────────────────────────────────────

PROCESS_REQUIRED_COLS = [
    'Batch_ID', 'Time_Minutes', 'Phase', 'Temperature_C', 'Pressure_Bar',
    'Humidity_Percent', 'Motor_Speed_RPM', 'Compression_Force_kN',
    'Flow_Rate_LPM', 'Power_Consumption_kW', 'Vibration_mm_s'
]

PRODUCTION_REQUIRED_COLS = [
    'Batch_ID', 'Granulation_Time', 'Binder_Amount', 'Drying_Temp',
    'Drying_Time', 'Compression_Force', 'Machine_Speed', 'Lubricant_Conc',
    'Moisture_Content', 'Tablet_Weight', 'Hardness', 'Friability',
    'Disintegration_Time', 'Dissolution_Rate', 'Content_Uniformity'
]


def run_pipeline(process_path: Path, production_path: Path,
                 verbose: bool = True) -> tuple:
    """
    Full data processing pipeline.
    Returns: (clean_process_df, clean_production_df, process_report, production_report)
    """
    # ── Process data ──
    p_report = DataQualityReport("batch_process_data")
    process_df = load_raw(process_path, p_report)
    process_df = normalize_column_names(process_df)
    process_df = validate_schema(process_df, PROCESS_REQUIRED_COLS, p_report)
    process_df = check_duplicates(process_df, p_report)
    process_df = handle_missing_values(process_df, p_report)
    process_df = detect_and_handle_outliers(process_df, p_report)
    process_df = validate_ranges(process_df, p_report)

    # ── Production data ──
    pr_report = DataQualityReport("batch_production_data")
    production_df = load_raw(production_path, pr_report)
    production_df = normalize_column_names(production_df)
    production_df = validate_schema(production_df, PRODUCTION_REQUIRED_COLS, pr_report)
    production_df = check_duplicates(production_df, pr_report)
    production_df = handle_missing_values(production_df, pr_report)
    production_df = detect_and_handle_outliers(production_df, pr_report)
    production_df = validate_ranges(production_df, pr_report)

    # ── Referential integrity: all production Batch_IDs should be valid ──
    process_batches = set(process_df['Batch_ID'].unique())
    production_batches = set(production_df['Batch_ID'].unique())
    unmatched = production_batches - process_batches
    if unmatched:
        pr_report.add_issue('INFO', 'Batch_ID',
                            f"{len(unmatched)} production batches have no process time-series "
                            f"(expected for multi-batch datasets)")

    if verbose:
        p_report.print_report()
        pr_report.print_report()

    return process_df, production_df, p_report, pr_report


def get_pipeline_summary(process_path: Path, production_path: Path) -> dict:
    """Returns pipeline quality report as dict — used by dashboard and API."""
    _, _, p_rep, pr_rep = run_pipeline(process_path, production_path, verbose=False)
    return {
        'process_data': p_rep.summary(),
        'production_data': pr_rep.summary(),
        'pipeline_status': 'PASS' if (
            p_rep.summary()['errors'] == 0 and
            pr_rep.summary()['errors'] == 0
        ) else 'FAIL'
    }


if __name__ == "__main__":
    PROCESS_PATH    = BASE_DIR / "data" / "_h_batch_process_data.xlsx"
    PRODUCTION_PATH = BASE_DIR / "data" / "_h_batch_production_data.xlsx"
    run_pipeline(PROCESS_PATH, PRODUCTION_PATH, verbose=True)