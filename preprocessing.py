"""
Preprocessing utilities for conversation data.

This module provides helper functions to normalise project identifiers,
filter data by recency and stitch messages into conversational turns.  It
also exposes a Spanish stopword list.
"""

import datetime
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Ensure the stopwords corpus is available.  The download is idempotent and
# will be skipped on subsequent runs.
nltk.download("stopwords", quiet=True)

# Expose a Spanish stopword list for reuse.  Some callers may choose to
# extend or override this list.
SPANISH_STOPWORDS = stopwords.words("spanish")

# Set of invalid or empty project names.  When normalising a comma separated
# list of projects any value in this set is ignored.
INVALID_PROJECT_VALUES = {"sin proyecto", "none", "null", "nan", ""}

def normalize_projects(val):
    """Split a project info string into a list of clean project identifiers.

    Empty or invalid values are removed and the remaining tokens are stripped
    of leading and trailing whitespace.

    Args:
        val: Raw project info value from a dataframe column.

    Returns:
        A list of project identifiers.
    """
    if val is None:
        return []
    text = str(val).strip()
    if text.lower() in INVALID_PROJECT_VALUES:
        return []
    return [p.strip() for p in text.split(",") if p.strip()]

def filter_last_n_months(df, date_col, months):
    """Return only the rows whose date is within the last *n* months.

    Args:
        df: Input dataframe with a datetime column.
        date_col: Name of the datetime column to use for filtering.
        months: Number of months to look back.

    Returns:
        A filtered dataframe containing only records newer than the cutoff.
    """
    max_date = df[date_col].max()
    if pd.isna(max_date):
        return df
    cutoff = max_date - pd.DateOffset(months=months)
    return df[df[date_col] >= cutoff]

def merge_turns(group):
    """Concatenate adjacent messages from the same sender into turns.

    Args:
        group: A dataframe containing messages for a single conversation, ordered
            by time.  Must contain 'sender' and 'text' columns.

    Returns:
        A single string with one line per turn: "sender: message".
    """
    turns = []
    last_sender = None
    buffer = []
    for _, row in group.iterrows():
        sender = row["sender"]
        text = str(row["text"]).strip()
        if sender != last_sender:
            if buffer:
                turns.append(f"{last_sender}: {' '.join(buffer)}")
                buffer = []
        buffer.append(text)
        last_sender = sender
    if buffer:
        turns.append(f"{last_sender}: {' '.join(buffer)}")
    return "\n".join(turns)

def merge_full_conversation(group):
    """Concatenate all messages in a conversation, preserving sender tags.

    Each message will be uppercased and prefixed with the sender.
    """
    messages = []
    for _, row in group.iterrows():
        sender = str(row["sender"]).upper()
        text = str(row["text"]).strip()
        if text:
            messages.append(f"{sender}: {text}")
    return "\n".join(messages)

def merge_turns_with_date(group):
    """Merge messages into turns and annotate them with the first message date.

    The returned string will include the date of the first message in each
    turn formatted as YYYY-MM-DD.
    """
    turns = []
    last_sender = None
    buffer = []
    buffer_dates = []
    group_sorted = group.sort_values("createdAt", ascending=False)
    for _, row in group_sorted.iterrows():
        sender = row["sender"]
        text = str(row["text"]).strip()
        fecha = row["createdAt"]
        if sender != last_sender:
            if buffer:
                date_str = buffer_dates[0].strftime("%Y-%m-%d")
                turns.append(f"{last_sender} ({date_str}): {' '.join(buffer)}")
                buffer = []
                buffer_dates = []
        buffer.append(text)
        buffer_dates.append(fecha)
        last_sender = sender
    if buffer:
        date_str = buffer_dates[0].strftime("%Y-%m-%d")
        turns.append(f"{last_sender} ({date_str}): {' '.join(buffer)}")
    return "\n".join(turns)